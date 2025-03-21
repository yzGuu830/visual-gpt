from .tokenizer import VisualTokenizer
from .cond_transformer import CondTransformer

from data import make_dl, make_inf_dl 
from utils import *


from tqdm import tqdm
import torch
from transformers import get_scheduler

import wandb

class Trainer:
    def __init__(self, arg, conf) -> None:
        self.arg = arg
        self.conf = conf

    def load(self, train_steps, warmup_steps):

        self.tokenizer = VisualTokenizer.from_pretrained(self.arg.from_pretrained)
        self.model = CondTransformer(self.tokenizer,
                                     condition_vocab=self.conf.transformer.num_classes,
                                     max_pos_len=self.conf.transformer.max_pos_len,
                                     d_model=self.conf.transformer.d_model,
                                     num_transformer_layers=self.conf.transformer.num_transformer_layers,
                                     num_attn_heads=self.conf.transformer.num_attn_heads
                                     )
        self.dls = make_dl(self.arg.data, train_bsz=self.arg.bsz, val_bsz=self.arg.bsz)
        self.opt = torch.optim.AdamW(self.model.lm.parameters(), lr=self.arg.lr, weight_decay=self.arg.wd)
        self.scheduler = get_scheduler(
            self.arg.scheduler,
            optimizer=self.opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )

    def train(self, train_steps=1000, warmup_steps=100, eval_every=300, device='cuda', save_path='./results/baseline_lm'):
        
        self.load(train_steps, warmup_steps)
        print(f"Number of parameters [Visual Tokenizer]: {sum(p.numel() for p in self.model.visual_tokenizer.parameters())}")
        print(f"Number of parameters [Transformer]: {sum(p.numel() for p in self.model.lm.parameters())}")

        self.model = self.model.to(device)
        self.model.lm.train()

        dl = make_inf_dl(self.dls['train'])
        pbar = tqdm(total=train_steps)
        losses = []
        for x, y in dl: 
            x, y = x.to(self.arg.device), y.to(self.arg.device)
            self.opt.zero_grad()
            loss, _ = self.model(x, cond=y.unsqueeze(1))
            loss.backward()
            self.opt.step()
            self.scheduler.step()
            pbar.update(1)

            pbar.set_description(f'[train step {pbar.n}] nll. loss: {loss.item():.4f} | ' + \
                                 f'ppl.: {torch.exp(loss).item():.4f}')
            
            if wandb.run is not None and pbar.n % 5 == 0:
                wandb.log({"nll_loss": loss.item(), 'ppl': torch.exp(loss).item()})
            
            losses.append(loss.item())
            if pbar.n % eval_every == 0 or pbar.n == train_steps:
                gens = self.eval_epoch(do_sample=True, device=device)
                vis_gens(gens, iteration=pbar.n+1, save_path=save_path)
                if pbar.n == train_steps: 
                    save_stats(losses, save_path)
                    save_model(self.model, save_path)
                    return
    
    def eval_epoch(self, do_sample=False, num_gen_imgs=1, device='cuda'):
        
        self.model.lm.eval()
        gens = []
        for i in range(self.conf.transformer.num_classes):
            cond = torch.full((num_gen_imgs, 1), i, dtype=torch.long, device=device)
            x_hat, _ = self.model.sample(cond, latent_size=tuple(self.conf.latent_size), do_sample=do_sample, temperature=1.2)
            gens.append(x_hat)

        gens = torch.stack(gens, dim=0) # [num_classes x num_gen_imgs x C x H x W]

        self.model.lm.train()
        return gens


    