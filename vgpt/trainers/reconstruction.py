import os
import torch
import tqdm
import json
import wandb
import numpy as np

from argparse import Namespace
from transformers import get_scheduler

from ..nn.tokenizers import (
    VisualTokenizer, 
    NLayerDiscriminator, 
    ReconLoss, 
    PerceptualLoss, 
    AdversarialLoss, 
    calculate_adaptive_weight
)

from .utils import (
    plot_reconstructions, 
    namespace2dict, 
    save_checkpoint, 
    load_checkpoint,
    img_normalize
)

from ..eval_metrics import (
    METRIC_FUNCS, VectorQuantEval, VectorQuantLatentVisualizer
)


class ReconTrainer:
    def __init__(self, conf: Namespace, arg: Namespace):
        self.model_conf = conf.model.tokenizer
        self.disc_conf = conf.model.discriminator
        self.exp_conf = conf.exp.recon
        
        self.arg = arg
        self.save_path = os.path.join(self.arg.save_path, 'tokenizer')


    def load_model(self):
        self.model = VisualTokenizer(self.model_conf.ae_name, 
                                     self.model_conf.qtz_name,
                                     namespace2dict(self.model_conf.ae_conf), 
                                     **namespace2dict(self.model_conf.vq_conf))
        self.model.to(self.arg.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model loaded!\nmodel # of params: {num_params/1e6}M')
        
        beta1, beta2 = self.exp_conf.betas
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.exp_conf.lr, betas=(beta1, beta2))
        print(f'Adam optimizer loaded with learning rate {self.exp_conf.lr} and betas ({beta1}, {beta2})\n')

        self.recon_loss_fn = ReconLoss(method=self.exp_conf.task_loss,)

        # load GANs if specified 
        if self.disc_conf is not None:
            self.discriminator = NLayerDiscriminator(**namespace2dict(self.disc_conf))
            self.discriminator.to(self.arg.device)
            num_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            print(f'discriminator loaded!\ndiscriminator # of params: {num_params/1e6}M')

            self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.exp_conf.lr, betas=(beta1, beta2))
            print(f'Adam optimizer loaded for discriminator with learning rate {self.exp_conf.lr} and betas ({beta1}, {beta2})')
            
            self.disc_start = self.exp_conf.disc_start
            print(f'discriminator training starts at step {self.disc_start}')
            self.adv_loss_fn = AdversarialLoss(self.discriminator, method='hinge')

        # load perceptual loss if specified
        self.pcpt_loss_fn = PerceptualLoss(self.arg.device)
        
        # load metrics 
        self.metric_fns = {}
        for m in self.exp_conf.metrics:
            m = m.strip().upper()
            if m not in METRIC_FUNCS:
                raise ValueError(f"Unknown metric '{m}' specified in config.")
            fn = METRIC_FUNCS[m]()
            if isinstance(fn, torch.nn.Module):
                fn.to(self.arg.device)
            self.metric_fns[m] = fn
        print(f'Loaded metrics: {", ".join(self.metric_fns.keys())}')


        self.vq_eval = VectorQuantEval(codebook_size=self.model.quantizer.num_codewords)
        if self.arg.latent_vis_every > 0: 
            self.latentvislzer = VectorQuantLatentVisualizer(save_dir=self.save_path, use_tsne=True, max_points=2048, use_wandb=(self.arg.wandb is not None))   


    def train_step(self, x, step=1):
        x = x.to(self.arg.device)
        disc_on = step >= self.disc_start if hasattr(self, 'disc_start') else False

        # generator update
        x_hat, vq_out = self.model(x)

        if 'cm_loss' in vq_out and 'cb_loss' in vq_out:
            cm_loss = vq_out.get('cm_loss').mean()
            cb_loss = vq_out.get('cb_loss').mean()
        else:
            cm_loss = torch.tensor(0.)
            cb_loss = torch.tensor(0.)

        recon_loss = self.recon_loss_fn(x_hat, x)
        p_loss = self.pcpt_loss_fn(x, x_hat).mean()
    
        if hasattr(self, 'discriminator') and disc_on:
            g_loss = self.adv_loss_fn.g_loss(x_hat.contiguous())
            d_weight = calculate_adaptive_weight(recon_loss, g_loss, self.exp_conf.disc_weight, 
                                        last_layer=self.model.get_last_layer())
        else:
            g_loss = torch.tensor(0.)
            d_weight = torch.tensor(0.)

        loss = recon_loss + \
                self.exp_conf.p_weight * p_loss + \
                self.exp_conf.disc_factor * d_weight * g_loss + \
                self.exp_conf.cm_weight * cm_loss + cb_loss 

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        # discriminator update
        if hasattr(self, 'discriminator') and disc_on:
            d_loss = self.exp_conf.disc_factor * self.adv_loss_fn.d_loss(x.contiguous().detach(), x_hat.contiguous().detach())
            self.disc_opt.zero_grad()
            d_loss.backward()
            self.disc_opt.step()
        else:
            d_loss = torch.tensor(0.)


        if vq_out.get('q') is not None:
            vq_active = vq_out.get('q').unique().numel() / self.model.quantizer.num_codewords
        else:
            vq_active = 0.
        if self.arg.latent_vis_every > 0 and (step-1) % self.arg.latent_vis_every == 0: 
            C = self.model.quantizer.codebook.data if self.model_conf.qtz_name == 'vq' else None
            self.latentvislzer.plot(z=vq_out.get('z_e'), 
                                    z_q=vq_out.get('z_q'),
                                    codebook=C,
                                    q=vq_out.get('q'),
                                    step=step
                                    )
            
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'p_loss': p_loss.item(),
            'cm_loss': cm_loss.item(),
            'cb_loss': cb_loss.item(),
            'vq_active': vq_active * 100,
            'g_loss': g_loss.item(),
            'd_weight': d_weight.item(),
            'd_loss': d_loss.item(),
        }


    def train(self, dataloaders):

        self.load_model()
        self.plt_batch_idx = np.random.randint(len(dataloaders.get('val'))-1)

        os.makedirs(self.save_path, exist_ok=True)
        json.dump(namespace2dict(self.model_conf), open(os.path.join(self.save_path, 'config.json'), 'w'), indent=4)

        self.model.train()
        pbar = tqdm.tqdm(total=self.exp_conf.train_steps)
        while True:
            for x, y in dataloaders.get('train'):
                log = self.train_step(x, step=pbar.n+1)
                pbar.update(1)

                desc = f'[train-step {pbar.n}/{self.exp_conf.train_steps}] ' + \
                          f'total loss: {log["loss"]:.4f} | ' + \
                          f'recon loss: {log["recon_loss"]:.4f} | ' + \
                          f'p loss: {log["p_loss"]:.4f} | ' + \
                          f'commit loss: {log["cm_loss"]:.4f} | ' + \
                          f'vq active ratio: {log["vq_active"]:.4f}%'
                if hasattr(self, 'discriminator') and pbar.n >= self.disc_start:
                    desc += f' | g loss: {log["g_loss"]:.4f} | d loss: {log["d_loss"]:.4f}'
                pbar.set_description(desc)

                if wandb.run is not None and pbar.n % self.exp_conf.log_interval == 0:
                    wandb.log(log, step=pbar.n)

                if pbar.n in self.exp_conf.checkpoint_steps:
                    save_checkpoint(self.model, self.save_path, f'model-ckpt-step-{pbar.n}.pth')

                if pbar.n % self.exp_conf.eval_interval == 0:
                    self.eval_step(dataloaders.get('val'), tag=f'training-step-{pbar.n}')

                if pbar.n == self.exp_conf.train_steps:
                    print("Training finished. Testing on validation set...\n--Final results--")
                    self.eval_step(dataloaders.get('val'), tag='final-step')
                    save_checkpoint(self.model, self.save_path, 'model.bin')
                    return


    @torch.no_grad()
    def eval_step(self, eval_loader, tag='train-step-100'):
        self.model.eval()
        
        self.vq_eval.reset_stats()
        stats = {m:[] for m in self.metric_fns.keys()} 
        for idx, (x, y) in tqdm.tqdm(enumerate(eval_loader), desc='Evaluating Model', total=len(eval_loader)): 
            x = x.to(self.arg.device)       # [0, 1] range
            x_hat, vq_out = self.model(x)

            x_hat = img_normalize(x_hat)    # normalize to [0, 1] range
            self.vq_eval.update(vq_out.get("q").long())
            for m, fn in self.metric_fns.items():
                stats[m].append(fn(x, x_hat))

            if idx == self.plt_batch_idx:
                x_np = x.cpu().numpy()
                x_hat_np = x_hat.cpu().numpy()
                x_np_plot, x_hat_np_plot = x_np.copy(), x_hat_np.copy()

        for m, vs in stats.items():
            stats[m] = np.mean(vs)

        vq_stats = self.vq_eval.cpt_stats()
        stats.update(vq_stats)
        print(f"{tag}: " + ' | '.join([f'{k}: {v:.5f}' for k, v in stats.items()]))
        
        plot_reconstructions(x_np_plot, x_hat_np_plot, tag, self.save_path, max_shown=20)
        
        if wandb.run is not None:
            wandb.log(stats)

        self.model.train()