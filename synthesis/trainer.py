from .tokenizer import VisualTokenizer
from .cond_transformer import CondTransformer

from data import make_dl, make_inf_dl 
from utils import *


from tqdm import tqdm
import torch
import argparse

from transformers import get_scheduler


class Trainer:
    def __init__(self, args) -> None:
        self.args = args

    def load(self, train_steps, warmup_steps):

        self.tokenizer = VisualTokenizer.from_pretrained(self.args.from_pretrained)
        self.model = CondTransformer(self.tokenizer,
                                     condition_vocab=self.args.num_classes,
                                     max_pos_len=1024,
                                     d_model=768,
                                     num_transformer_layers=12,
                                     num_attn_heads=12,)
        self.dls = make_dl(self.args.data, train_bsz=self.args.bsz, val_bsz=self.args.bsz)
        self.opt = torch.optim.AdamW(self.model.lm.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.scheduler = get_scheduler(
            self.args.scheduler,
            optimizer=self.opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )

    def train(self, train_steps=1000, warmup_steps=100, eval_every=300, device='cuda', save_path='./results/baseline_lm'):
        
        self.load(train_steps, warmup_steps)
        print(f"Number of parameters [VQ-VAE]: {sum(p.numel() for p in self.model.visual_tokenizer.parameters())}")
        print(f"Number of parameters [Transformer]: {sum(p.numel() for p in self.model.lm.parameters())}")

        self.model = self.model.to(device)
        self.model.lm.train()

        dl = make_inf_dl(self.dls['train'])
        pbar = tqdm(total=train_steps)
        losses = []
        for x, y in dl: 
            x, y = x.to(self.args.device), y.to(self.args.device)
            self.opt.zero_grad()
            loss, _ = self.model(x, cond=y.unsqueeze(1))
            loss.backward()
            self.opt.step()
            self.scheduler.step()
            pbar.update(1)

            pbar.set_description(f'[train step {pbar.n}] nll. loss: {loss.item():.4f} | ' + \
                                 f'ppl.: {torch.exp(loss).item():.4f}')
            
            losses.append(loss.item())
            
            if pbar.n % eval_every == 0 or pbar.n == train_steps:
                gens = self.eval_epoch(do_sample=True, device=device)
                vis_gens(gens, iteration=pbar.n+1, save_path=save_path)
                if pbar.n == train_steps: 
                    save_stats(losses, save_path)
                    save_model(self.model, save_path)
                    return
    
    def eval_epoch(self, do_sample=False, device='cuda'):
        
        self.model.lm.eval()

        gens = []
        for i in range(self.args.num_classes):
            cond = torch.full((1, 1), i, dtype=torch.long, device=device)
            x_hat, _ = self.model.sample(cond, latent_size=(8,8), do_sample=do_sample)
            gens.append(x_hat)

        gens = torch.cat(gens, dim=0)

        self.model.lm.train()
        return gens
    

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--from_pretrained', type=str, default='outputs/resnet_cifar10_baseline')
    parser.add_argument('--output_path', type=str, default='./results')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_name', type=str, default='baseline', help='experiment name')
    
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)

    return parser.parse_args()

def run():
    args = parse_args()
    seed_everything(args.seed)

    trainer = Trainer(args)

    trainer.train(train_steps=args.train_steps, warmup_steps=args.warmup_steps, eval_every=args.eval_every, 
                  device=args.device, save_path=os.path.join(args.output_path, args.exp_name))


    