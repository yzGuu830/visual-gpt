import torch
import torch.nn as nn

import math

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class CondTransformer(nn.Module):
    def __init__(self, 
                 visual_tokenizer,
                 condition_vocab=10,
                 max_pos_len=1024,
                 d_model=768,
                 num_transformer_layers=12,
                 num_attn_heads=12,
                 ):
        super().__init__()
        
        self.visual_tokenizer = visual_tokenizer
        self.visual_tokenizer.eval()

        self.latent_vocab = visual_tokenizer.model.quantizer.num_codewords
        config = GPT2Config(
            vocab_size=self.latent_vocab+condition_vocab+1,
            n_positions=max_pos_len,
            n_embd=d_model,
            n_layer=num_transformer_layers,
            n_head=num_attn_heads,
            bos_token_id=self.latent_vocab+condition_vocab, 
            eos_token_id=self.latent_vocab+condition_vocab
        )
        self.lm = GPT2LMHeadModel(config)
        self.config = {
            'condition_vocab': condition_vocab,
            'max_pos_len': max_pos_len,
            'd_model': d_model,
            'num_transformer_layers': num_transformer_layers,
            'num_attn_heads': num_attn_heads,
        }
    
    def forward(self, x, cond=None):
        """
        Args:
            x (torch.Tensor): input img of shape (bsz, C, H, W)
            cond (torch.Tensor): condition (class) code of shape (bsz, 1)
        """
        z = self.encode_to_z(x) # [bsz x HW]
        
        if cond is None: # unconditional
            c = torch.full((x.size(0), 1), self.lm.config.bos_token_id, dtype=torch.long, device=x.device)
        else:
            c = self.encode_to_c(cond)
        num_c_tokens = c.size(1)
        # print("example c:", c[0].tolist())
        # print("example z:", z[0].tolist())

        # concatenate latent code and condition code
        in_seq = torch.cat([c, z], dim=-1) # [bsz x (HW+1)]
        outputs = self.lm(input_ids=in_seq[:, :-1], return_dict=True)

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
        if cond.size(1) == 1: # class label
            cond = cond + self.visual_tokenizer.model.quantizer.num_codewords # start from latent vocab
            return cond
        else:
            raise NotImplementedError("Not implemented for condition of other modalities")
    
    @torch.no_grad()
    def sample(self, cond, latent_size=(4,4), temperature=1.0, top_k=None, do_sample=False, num_gen_imgs=1):
        """
        Args:
            cond (torch.Tensor): condition (class) code of shape (bsz, 1)
            latent_size (int): size of latent code
        """
        if cond is None: # unconditional
            c = torch.full((num_gen_imgs, 1), self.lm.config.bos_token_id, dtype=torch.long, device=self.lm.device)
        else:
            c = self.encode_to_c(cond)
        
        # greedy decoding
        z = c
        for _ in range(math.prod(latent_size)):

            outputs = self.lm(input_ids=z, return_dict=True)
            lm_logits = outputs.logits[:, -1, :]
            lm_logits = lm_logits / temperature
        
            if top_k is not None:
                v, _ = torch.topk(lm_logits, top_k)
                lm_logits[lm_logits < v[..., [-1]]] = float('-inf')

            probs = nn.functional.softmax(lm_logits, dim=-1)
            if do_sample:
                probs[:, self.latent_vocab:] = 0 # mask condition and eos/bos tokens
                dec_token = torch.multinomial(probs, num_samples=1)
            else:
                probs[:, self.latent_vocab:] = float('-inf') # mask condition and eos/bos tokens
                _, dec_token = torch.topk(probs, k=1, dim=-1)

            z = torch.cat((z, dec_token), dim=1)
        # outputs = self.lm.generate(
        #     input_ids = z,
        #     do_sample = do_sample,
        #     max_length = math.prod(latent_size) + 1,
        #     min_length = math.prod(latent_size) + 1,
        #     temperature = temperature,
        #     top_k = top_k,
        # )
        # print('model.generate(): ', outputs)
        # 1/0

        z = z[:, c.size(1):]
        x_hat, code = self.reconstruct_from_code(z, latent_size)
        return x_hat, code
    
    @torch.no_grad()
    def reconstruct_from_code(self, code, latent_size=(4,4)):
        input_shape = (-1,) + latent_size
        code = code.view(input_shape)
        x_hat = self.visual_tokenizer.decode(code)
        return x_hat, code


if __name__ == "__main__":

    from tokenizer import VisualTokenizer
    tokenizer = VisualTokenizer.from_pretrained("results/resnet_cifar10_large")

    model = CondTransformer(
        visual_tokenizer=tokenizer,
        condition_vocab=10,
        max_pos_len=1024,
        d_model=64,
        num_transformer_layers=4,
        num_attn_heads=2,
    )

    x = torch.rand(2, 3, 32, 32)
    cond = torch.tensor([[1], 
                         [2]])
    loss, lm_logits = model(x, cond)
    print(f"Loss: {loss}, Logits: {lm_logits.shape}")

    x_hat, code = model.sample(cond, latent_size=(8,8), do_sample=False)
    print(x_hat.shape)
    print(code.shape, code)