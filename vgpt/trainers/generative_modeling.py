import os
import torch
import tqdm
import json
import wandb

from argparse import Namespace
from transformers import get_scheduler

from ..nn.tokenizers import VisualTokenizer
from ..nn.generators import CondVisualGPT

from .utils import (
    plot_generations, 
    namespace2dict, 
    save_checkpoint, 
    load_checkpoint
)

from ..eval_metrics import METRIC_FUNCS



class GenTrainer:
    def __init__(self, conf: Namespace, arg: Namespace):
        self.data_conf = conf.data
        self.model_conf = conf.model.transformer
        self.exp_conf = conf.exp.ar
        
        self.arg = arg
        self.save_path = os.path.join(self.arg.save_path, 'gpt')


    def load_model(self, tokenizer_path):
        visual_tokenizer = VisualTokenizer.from_pretrained(os.path.join(tokenizer_path, "tokenizer"))
        self.model = CondVisualGPT(visual_tokenizer, **namespace2dict(self.model_conf))
        self.model.to(self.arg.device)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model loaded!\nmodel # of params: {num_params/1e6}M')

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.exp_conf.lr, weight_decay=self.exp_conf.wd, 
                                     betas=tuple(self.exp_conf.betas))
        print(f'AdamW optimizer loaded with learning rate {self.exp_conf.lr}, weight decay {self.exp_conf.wd} and betas {tuple(self.exp_conf.betas)})')
        
        self.scheduler = get_scheduler(
            self.exp_conf.scheduler,
            optimizer=self.opt,
            num_warmup_steps=self.exp_conf.warmup_steps,
            num_training_steps=self.exp_conf.train_steps,
        )
        print(f'{self.exp_conf.scheduler} scheduler loaded with {self.exp_conf.warmup_steps/1e3}K warmup steps ')

    def train_step(self, x, y):
        x, y = x.to(self.arg.device), y.to(self.arg.device)

        loss, _ = self.model(x, cond=y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.scheduler.step()

        return {
            'nll_loss': loss.item(),
            'ppl': torch.exp(loss).item()
        }

    def train(self, dataloaders, tokenizer_path):

        self.load_data()
        self.load_model(tokenizer_path)

        os.makedirs(self.save_path, exist_ok=True)
        json.dump(namespace2dict(self.model_conf), open(os.path.join(self.save_path, 'config.json'), 'w'), indent=4)

        self.model.train()
        pbar = tqdm.tqdm(total=self.exp_conf.train_steps)
        while True:
            for x, y in dataloaders.get('train'):
                if y.dim() == 1: y.unsqueeze_(1)
                log = self.train_step(x, y)
                pbar.update(1)

                desc = f'[train-step {pbar.n}/{self.exp_conf.train_steps}] ' + \
                          f'nll. loss: {log["nll_loss"]:.4f} | ' + \
                          f'ppl.: {log["ppl"]:.4f}'
                pbar.set_description(desc)

                if wandb.run is not None and pbar.n % self.exp_conf.log_interval == 0:
                    wandb.log(log, step=pbar.n)

                if pbar.n % self.exp_conf.eval_interval == 0:
                    self.eval_step(num_class=5, num_sample=10, tag="LM_Generated_Imgs@Iteration{}".format(pbar.n))

                if pbar.n in self.exp_conf.checkpoint_steps:
                    save_checkpoint(self.model.gpt, self.save_path, f'model-ckpt-step-{pbar.n}.pth')

                if pbar.n == self.exp_conf.train_steps:
                    print("Training finished. Saving final model...")
                    save_checkpoint(self.model.gpt, self.save_path, 'model.bin')
                    self.eval_step(num_class=5, num_sample=10, tag="LM_Generated_Imgs@Final")
                    return
    
    @torch.no_grad()
    def eval_step(self, num_class, num_sample, tag):

        if num_class > self.model_conf.condition_vocab:
            raise ValueError(f"num_class {num_class} exceeds the condition vocab size {self.model_conf.condition_vocab}.")
        
        self.model.gpt.eval()
        imgs = {} 
        for i in range(num_class):
            cond = torch.full((1, 1), i, dtype=torch.long, device=self.arg.device)
            x_hat, _ = self.model.sample(cond, z_shape=tuple(self.exp_conf.z_shape), num_return_sequences=num_sample, 
                                         do_sample=True, temperature=1.0, top_k=100)

            imgs[f'class{i}'] = x_hat # (num_sample, C, H, W)

        plot_generations(imgs, tag, save_path=self.save_path)
        self.model.gpt.train()
        return imgs