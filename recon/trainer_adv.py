import wandb

import torch.nn.functional as F
from tqdm import tqdm

from .trainer import Trainer
from .models import load_model, NLayerDiscriminator
from .losses import d_loss, g_loss

from vector_quantize import freeze_vq_forward_hook
from data import make_dl, make_inf_dl
from utils import *


class TrainerAdv(Trainer):
    def __init__(self, arg, conf):
        super(TrainerAdv, self).__init__(arg, conf)
    
    def load(self, ):

        self.dls = make_dl(self.arg.data_name, self.conf.exp.bsz, self.conf.exp.bsz, **namespace2dict(self.conf.data))
        
        self.model = load_model(self.conf.model)
        self.model.to(self.arg.device)

        if self.conf.exp.pretrain_steps > 0:
            self.model.quantizer.register_buffer('is_freezed', torch.ones(1))
            self.model.quantizer.register_forward_hook(freeze_vq_forward_hook)
            print(f'pretraining autoencoder for {self.conf.exp.pretrain_steps} steps')
        else:
            self.model.quantizer.register_buffer('is_freezed', torch.zeros(1))

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model Loaded!\nModel # of Params: {num_params}')

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.conf.exp.lr)
        print(f'Optimizer: Adam\nLearning rate: {self.conf.exp.lr}\n')

        self.discriminator = NLayerDiscriminator(**namespace2dict(self.conf.discriminator))
        self.discriminator.to(self.arg.device)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.conf.exp.lr)
        num_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f'Discriminator Loaded!\nDiscriminator # of Params: {num_params}')
        self.disc_start = self.conf.exp.disc_start
        return 
    

    def train(self, ):
        self.load()
        self.model.train()

        if not os.path.exists(self.arg.save_path): os.makedirs(self.arg.save_path)
        # f = open(self.arg.save_path + '/log.txt', 'w')
        f = None

        pbar = tqdm(total=self.conf.exp.steps)
        dl = make_inf_dl(self.dls['train'])
        for x, y in dl: 
            x, y = x.to(self.arg.device), y.to(self.arg.device)

            x_hat, vq_out = self.model(x)

            recon_loss = F.mse_loss(x_hat, x)
            vq_loss = self.conf.exp.beta * vq_out['cm_loss'].mean() + vq_out['cb_loss'].mean()
            loss = recon_loss + vq_loss
            
            if pbar.n >= self.conf.exp.disc_start:
                logits_fake = self.discriminator(x_hat.contiguous())
                gen_loss = g_loss(logits_fake)
                loss += gen_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if pbar.n >= self.conf.exp.disc_start:
                self.discriminator_opt.zero_grad()
                logits_fake = self.discriminator(x_hat.contiguous().detach())
                logits_real = self.discriminator(x.contiguous().detach())
                disc_loss = d_loss(logits_real, logits_fake, method='vanilla')
                disc_loss.backward()
                self.discriminator_opt.step()
            
            pbar.update(1)

            
            if self.model.quantizer.is_freezed.item() == 0:
                active_ratio = vq_out['q'].unique().numel() / self.conf.model.num_codewords
            else:
                active_ratio = 0
            desc = f'[Train step {pbar.n}/{self.conf.exp.steps}] total loss: {loss.item():.4f} | ' + \
                   f'recon loss: {recon_loss.item():.4f} | ' + \
                   f'vq loss: {vq_loss.item():.4f} | ' + \
                   f'vq active ratio: {active_ratio*100:.4f}%'
            if pbar.n-1 >= self.conf.exp.disc_start:
                desc += f' | disc loss: {disc_loss.item():.4f} | gen loss: {gen_loss.item():.4f}'
            pbar.set_description(desc)
            # f.write(desc + '\n')

            if wandb.run is not None and pbar.n % self.conf.exp.log_interval == 0: 
                log_stats = {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'vq_loss': vq_loss.item(), 'vq_active_ratio': active_ratio}
                if pbar.n-1 >= self.conf.exp.disc_start:
                    log_stats.update({'disc_loss': disc_loss.item(), 'gen_loss': gen_loss.item()})
                wandb.log(log_stats)

            if pbar.n > self.conf.exp.pretrain_steps and pbar.n % self.conf.exp.eval_interval == 0:
                print(f'[Test step {pbar.n+1}/{self.conf.exp.steps}]... ', end='', file=f)
                self.eval_epoch(tag=f'training-step-{pbar.n}', f=f)
                self.model.train()

            if pbar.n == self.conf.exp.steps:
                print("Training finished. Testing on validation set...\n--Final results--", file=f)
                self.eval_epoch(tag='final', f=f)
                save_checkpoint(self.model, self.arg.save_path)
                return
            
            if pbar.n == self.conf.exp.pretrain_steps:
                self.model.quantizer.is_freezed.fill_(0)
                print(f'Activating VQ layer after {self.conf.exp.pretrain_steps} steps') 