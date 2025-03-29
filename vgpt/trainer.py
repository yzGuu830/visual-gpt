import os
import tqdm
import json
import wandb

from argparse import Namespace
from transformers import get_scheduler

from .tokenizer import VisualTokenizer, NLayerDiscriminator, ReconLoss, PerceptualLoss, AdversarialLoss, calculate_adaptive_weight
from .generator import CondVisualGPT

from dataset.data import make_dl

from .utils import *
from utils import *


class ReconTrainer:
    def __init__(self, conf: Namespace, arg: Namespace):
        self.data_conf = conf.data
        self.model_conf = conf.model.tokenizer
        self.disc_conf = conf.model.discriminator
        self.exp_conf = conf.exp.recon
        
        self.arg = arg
        self.save_path = os.path.join(self.arg.save_path, 'tokenizer')

    def load_data(self):
        self.dls = make_dl(train_bsz=self.exp_conf.bsz, val_bsz=self.exp_conf.bsz, **namespace2dict(self.data_conf))

    def load_model(self):
        self.model = VisualTokenizer(self.model_conf.ae_name, 
                                     namespace2dict(self.model_conf.ae_conf), 
                                     **namespace2dict(self.model_conf.vq_conf))
        self.model.to(self.arg.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model loaded!\nmodel # of params: {num_params/1e6}M')
        
        beta1, beta2 = 0.5, 0.9
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.exp_conf.lr, betas=(beta1, beta2))
        print(f'Adam optimizer loaded with learning rate {self.exp_conf.lr} and betas ({beta1}, {beta2})\n')

        self.recon_loss_fn = ReconLoss(method='l2' if self.disc_conf is None else 'l1')

        if self.disc_conf is not None:
            self.discriminator = NLayerDiscriminator(**namespace2dict(self.disc_conf))
            self.discriminator.to(self.arg.device)
            num_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            print(f'discriminator loaded!\ndiscriminator # of params: {num_params/1e6}M')

            beta1, beta2 = 0.5, 0.9
            self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.exp_conf.lr, betas=(beta1, beta2))
            print(f'Adam optimizer loaded for discriminator with learning rate {self.exp_conf.lr} and betas ({beta1}, {beta2})')
            
            self.disc_start = self.exp_conf.disc_start
            print(f'discriminator training starts at step {self.disc_start}')
            self.adv_loss_fn = AdversarialLoss(self.discriminator, method='hinge')
        
        if self.exp_conf.p_weight > 0:
            self.pcpt_loss_fn = PerceptualLoss(self.arg.device)
        
        self.metric_fns = {}
        for m in self.exp_conf.metrics:
            m = m.strip().lower()
            if m not in METRICS:
                raise ValueError(f"Unknown metric '{m}' specified in config.")
            self.metric_fns[m] = METRICS[m]
        self.cnter = VQCodebookCounter(codebook_size=self.model.quantizer.num_codewords, device=self.arg.device)    

    def train_step(self, x, disc_on=False):
        x = x.to(self.arg.device)

        # generator update
        x_hat, vq_out = self.model(x)

        cm_loss = vq_out.get('cm_loss').mean()
        cb_loss = vq_out.get('cb_loss').mean()

        recon_loss = self.recon_loss_fn(x_hat, x)
        
        if hasattr(self, 'pcpt_loss_fn'):
            p_loss = self.pcpt_loss_fn(x, x_hat).mean()
        else:
            p_loss = torch.tensor(0.)
    
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
            
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'p_loss': p_loss.item(),
            'cm_loss': cm_loss.item(),
            'cb_loss': cb_loss.item(),
            'g_loss': g_loss.item(),
            'd_weight': d_weight.item(),
            'd_loss': d_loss.item(),
            'vq_active': vq_active * 100,
        }

    def train(self, ):

        self.load_data()
        self.load_model()

        os.makedirs(self.save_path, exist_ok=True)
        json.dump(namespace2dict(self.model_conf), open(os.path.join(self.save_path, 'config.json'), 'w'), indent=4)

        self.model.train()
        pbar = tqdm.tqdm(total=self.exp_conf.train_steps)
        while True:
            for x, y in self.dls.get('train'):
                log = self.train_step(x, disc_on=pbar.n >= self.disc_start)
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
                    self.eval_step(tag=f'training-step-{pbar.n}')

                if pbar.n == self.exp_conf.train_steps:
                    print("Training finished. Testing on validation set...\n--Final results--")
                    self.eval_epoch(tag='final-step')
                    save_checkpoint(self.model, self.save_path, 'model.pth')
                    return

    @torch.no_grad()
    def eval_step(self, tag='train-step-100'):
        self.model.eval()
        
        plt_batch_idx = np.random.randint(len(self.dls.get('val'))-1)
        self.cnter.reset_stats(1)
        stats = {m:[] for m in self.metric_fns.keys()} 
        for idx, (x, y) in enumerate(self.dls.get('val')): 
            x = x.to(self.arg.device)

            x_hat, vq_out = self.model(x)
            self.cnter.update(vq_out.get("q")[:, None, None, :])

            x_np = x.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()

            for i in range(x_np.shape[0]):
                for m, fn in self.metric_fns.items():
                    stats[m].append(fn(x_np[i], x_hat_np[i]))

            if idx == plt_batch_idx:
                x_np_plot, x_hat_np_plot = x_np.copy(), x_hat_np.copy()

        for m, vs in stats.items():
            stats[m] = sum(vs) / len(vs)
        util_ratio, _ = self.cnter.compute_utilization()
        stats['util_ratio'] = util_ratio*100
        print(f"{tag}: " + ' | '.join([f'{k}: {v:.5f}' for k, v in stats.items()]))
        
        plot_reconstructions(x_np_plot, x_hat_np_plot, tag, self.save_path)
        if wandb.run is not None:
            wandb.log(stats)

        self.model.train()


class GenTrainer:
    def __init__(self, conf: Namespace, arg: Namespace):
        self.data_conf = conf.data
        self.model_conf = conf.model.transformer
        self.exp_conf = conf.exp.ar
        
        self.arg = arg
        self.save_path = os.path.join(self.arg.save_path, 'gpt')

    def load_data(self):
        self.dls = make_dl(train_bsz=self.exp_conf.bsz, val_bsz=self.exp_conf.bsz, **namespace2dict(self.data_conf))

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

    def train(self, tokenizer_path):

        self.load_data()
        self.load_model(tokenizer_path)

        os.makedirs(self.save_path, exist_ok=True)
        json.dump(namespace2dict(self.model_conf), open(os.path.join(self.save_path, 'config.json'), 'w'), indent=4)

        self.model.train()
        pbar = tqdm.tqdm(total=self.exp_conf.train_steps)
        while True:
            for x, y in self.dls.get('train'):
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
                    save_checkpoint(self.model, self.save_path, f'model-ckpt-step-{pbar.n}.pth')

                if pbar.n == self.exp_conf.train_steps:
                    print("Training finished. Saving final model...")
                    save_checkpoint(self.model, self.save_path, 'model.pth')
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