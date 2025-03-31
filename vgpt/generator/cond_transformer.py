import torch
import torch.nn as nn
import os

import math
import json

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from ..tokenizer import VisualTokenizer



class CondVisualGPT(nn.Module):
    def __init__(self, 
                 visual_tokenizer: VisualTokenizer,
                 condition_vocab=10,
                 max_pos_len=1024,
                 d_model=768,
                 num_transformer_layers=12,
                 num_attn_heads=12,
                 ):
        super().__init__()
        
        self.visual_tokenizer = visual_tokenizer
        self.visual_tokenizer.eval()

        self.latent_vocab = visual_tokenizer.quantizer.num_codewords
        config = GPT2Config(
            vocab_size=self.latent_vocab+condition_vocab+2,
            n_positions=max_pos_len,
            n_embd=d_model,
            n_layer=num_transformer_layers,
            n_head=num_attn_heads,
            bos_token_id=self.latent_vocab+condition_vocab, 
            eos_token_id=self.latent_vocab+condition_vocab,
        )
        self.gpt = GPT2LMHeadModel(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str="Imagenet100-VQCondTransformer"):

        visual_tokenizer = VisualTokenizer.from_pretrained(os.path.join(pretrained_model_name_or_path, "tokenizer"))
        
        gpt_config = json.load(open(os.path.join(pretrained_model_name_or_path, 'gpt', 'config.json'), "r"))
        visual_gpt = cls(visual_tokenizer, **gpt_config)
        visual_gpt.gpt.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, 'gpt', 'model.bin'), map_location="cpu", weights_only=True), strict=False)
        
        visual_gpt.id2label = json.load(open(os.path.join(pretrained_model_name_or_path, 'id2label.json'), "r"))
        
        print("visual autoregressive transformer loaded from pretrained {}!".format(os.path.join(pretrained_model_name_or_path, "gpt")))
        return visual_gpt
    
    def forward(self, x, cond=None):
        """
        To perform conditional generations, simply prepend condition token (e.g., class, text, image) to the image sequence.
        To perform unconditional generations, simply pass `cond=None` and will use the bos_token_id as the start token. 
        
        Args:
            x (torch.Tensor): input img of shape (bsz, C, H, W)
            cond (torch.Tensor): condition (class) code of shape (bsz, 1)
        Returns:
            loss (torch.Tensor): nll loss from next token prediction
            lm_logits (torch.Tensor): output logits from the transformer model (including condition logits)
        """
        z = self.encode_to_z(x) # [bsz x HW]
        
        if cond is None: # unconditional
            c = torch.full((x.size(0), 1), self.gpt.config.bos_token_id, dtype=torch.long, device=x.device)
        else:
            c = self.encode_to_c(cond)
        num_c_tokens = c.size(1)

        # concatenate latent code and condition code
        in_seq = torch.cat([c, z], dim=-1) # [bsz x (HW+1)]
        outputs = self.gpt(input_ids=in_seq[:, :-1], return_dict=True)

        lm_logits = outputs.logits
        z_logits = lm_logits[:, num_c_tokens-1:].contiguous()
        loss = nn.functional.cross_entropy(z_logits.view(-1, z_logits.size(-1)), z.view(-1))

        return loss, lm_logits

    @torch.no_grad()
    def encode_to_z(self, x):
        z = self.visual_tokenizer.encode(x) # bsz x HW
        return z
    
    @torch.no_grad()
    def encode_to_c(self, cond):
        return cond + self.visual_tokenizer.quantizer.num_codewords # start from latent vocab

    @torch.no_grad()
    def sample(self, cond, z_shape=(16,16), **gen_kwargs):
        """
        Given a condition code, perform autoregressive sampling

        Args:
            cond (torch.Tensor): condition (class) code of shape (bsz, 1)
            z_shape (tuple): latent resolution h, w
            gen_kwargs: additional arguments for huggingface's model.generate()
        Returns:
            x_hat (torch.Tensor): generated image of shape (bsz, C, H, W)
            code (torch.Tensor): generated latent code of shape (bsz, h, w)
        """
        if cond is None: # unconditional
            c = torch.full((1, 1), self.bos_token_id, dtype=torch.long, device=self.gpt.device)
        else:
            c = self.encode_to_c(cond)

        output_ids = self.gpt.generate(
            input_ids = c,
            min_new_tokens = math.prod(z_shape),
            max_new_tokens = math.prod(z_shape),
            bad_words_ids = self.condition_token_ids, # avoid generating condition tokens
            pad_token_id = self.unk_token_id,
            **gen_kwargs
        ) # (bsz, hw)
        output_ids = output_ids[:, c.size(1):]
        x_hat, code = self.reconstruct_from_code(output_ids, z_shape)
        return x_hat, code
    
    @torch.no_grad()
    def reconstruct_from_code(self, code, z_shape=(16,16)):
        input_shape = (-1,) + z_shape
        code = code.view(input_shape)
        x_hat = self.visual_tokenizer.decode(code)
        return x_hat, code
    
    @property
    def visual_token_ids(self):
        return [[i] for i in range(self.visual_tokenizer.quantizer.num_codewords)]
    
    @property
    def condition_token_ids(self):
        return [[i] for i in range(self.visual_tokenizer.quantizer.num_codewords, self.gpt.config.vocab_size)]
    
    @property
    def bos_token_id(self):
        return self.gpt.config.bos_token_id
    
    @property
    def unk_token_id(self):
        return self.gpt.config.bos_token_id + 1 # unused for handling pad_token
    
    def img_normalize(self, batch):
        # batch: (B, C, H, W)
        B = batch.size(0)
        
        flat = batch.contiguous().view(B, -1)
        mins = flat.min(dim=1)[0].view(B, 1, 1, 1)
        maxs = flat.max(dim=1)[0].view(B, 1, 1, 1)

        normalized = (batch - mins) / (maxs - mins + 1e-8)
        return normalized
    



from transformers import ViTForImageClassification, ViTImageProcessor
 
class ClassConditionedRejectionSampler:
    def __init__(self, 
                 model: CondVisualGPT, 
                 classifier: ViTForImageClassification,
                 processor: ViTImageProcessor):
        self.model = model
        self.classifier = classifier

        self.model.eval()
        self.classifier.eval()

        self.processor = processor

    @torch.no_grad()
    def sample(self, cls_name: str, z_shape=(16,16), accept_n=1, **gen_kwargs):
        """
        Perform class-conditional sampling with classifier-rejection.
        Args:
            cls_name (str): class name for condition
            z_shape (tuple): shape of the generated latent code (h, w)
            accept_n (int): select best from `accept_n` samples based on classifier score.
            gen_kwargs: additional arguments for the model.generate() method
        """
        print(f"rejection sampling {accept_n} images for class: {cls_name}...")
        cond = torch.full((1,1), self.label2id_generator(cls_name), dtype=torch.long, device=self.model.gpt.device)
        x_hat, code = self.model.sample(cond, z_shape=z_shape, num_return_sequences=accept_n, **gen_kwargs) # generated image and code

        scores = self.compute_class_score(self.model.img_normalize(x_hat), cls_name)
        b_idx = scores.argmax().item()
        print(f"done! rejection sampling best 1/{accept_n}: best image score is {scores[b_idx]:.4f}")
        return x_hat[b_idx:b_idx+1], code[b_idx:b_idx+1], scores[b_idx:b_idx+1]
    

    def compute_class_score(self, x: torch.Tensor, cls_name: str):

        class_id = self.label2id_classifier(cls_name)
        inputs = self.processor(images=x, return_tensors="pt")
        outputs = self.classifier(**inputs)
        logits = outputs.logits

        score = torch.nn.functional.softmax(logits, dim=1)[:, class_id]
        return score
    

    def label2id_classifier(self, cls_name: str):
        id2label = self.classifier.config.id2label
        if cls_name in id2label.values():
            class_id = next(k for k, v in id2label.items() if v == cls_name)
        else:
            class_id = next((k for k, v in id2label.items() if v.startswith(cls_name)), None) # handle label difference

        if class_id is None:
            raise ValueError(f"Class name '{cls_name}' not found in {type(self.classifier)} classifier's id2label mapping.")
        return int(class_id)
    

    def label2id_generator(self, cls_name: str):
        id2label = self.model.id2label
        if cls_name in id2label.values():
            class_id = next(k for k, v in id2label.items() if v == cls_name)
        else:
            class_id = next((k for k, v in id2label.items() if v.startswith(cls_name)), None)
        
        if class_id is None:    
            raise ValueError(f"Class name '{cls_name}' not found in the generator's id2label mapping.")
        return int(class_id)
    

    