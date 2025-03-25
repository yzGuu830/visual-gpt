import wandb
import json
import torch.nn.functional as F
from tqdm import tqdm

from .models import load_model
from .utils import *

from data import make_dl, make_inf_dl
from utils import *

from vector_quantize import freeze_dict_forward_hook


class Trainer:

    def __init__(self, arg, conf):
        
        self.arg = arg
        self.conf = conf

    def load(self, ):

        self.dls = make_dl(self.conf.data.data_name, self.conf.exp.bsz, self.conf.exp.bsz, **namespace2dict(self.conf.data.dl_kwargs))
        
        self.model = load_model(self.conf.model)
        self.model.to(self.arg.device)

        if self.conf.exp.pretrain_steps > 0:
            self.model.quantizer.register_buffer('is_freezed', torch.ones(1))
            self.model.quantizer.register_forward_hook(freeze_dict_forward_hook)
            print(f'pretraining autoencoder for {self.conf.exp.pretrain_steps} steps')
        else:
            self.model.quantizer.register_buffer('is_freezed', torch.zeros(1))

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model Loaded!\nModel # of Params: {num_params/1e6}M')

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.conf.exp.lr)
        print(f'Optimizer: Adam\nLearning rate: {self.conf.exp.lr}\n')
        return 


    def train(self, ):
        self.load()
        self.model.train()

        if not os.path.exists(self.arg.save_path): os.makedirs(self.arg.save_path)
        json.dump(namespace2dict(self.conf.model), open(self.arg.save_path + '/config.json', 'w'), indent=4)
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

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            pbar.update(1)

            if self.model.quantizer.is_freezed.item() == 0:
                active_ratio = vq_out['q'].unique().numel() / self.conf.model.vq.num_codewords
            else:
                active_ratio = 0
            desc = f'[Train step {pbar.n}/{self.conf.exp.steps}] total loss: {loss.item():.4f} | ' + \
                   f'recon loss: {recon_loss.item():.4f} | ' + \
                   f'vq loss: {vq_loss.item():.4f} | ' + \
                   f'vq active ratio: {active_ratio*100:.4f}%'
            pbar.set_description(desc)
            # f.write(desc + '\n')

            if wandb.run is not None and pbar.n % self.conf.exp.log_interval == 0: 
                log_stats = {'loss': loss.item(), 'recon_loss': recon_loss.item(), 'vq_loss': vq_loss.item(), 'vq_active_ratio': active_ratio}
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
            
            
    @torch.no_grad()
    def eval_epoch(self, tag='', f=None):
        self.model.eval()
        logs = {m:[] for m in METRIC_FUNCS.keys()} 

        e_counter = VQCodebookCounter(codebook_size=self.conf.model.vq.num_codewords, device=self.arg.device)
        e_counter.reset_stats(1)
        for x, y in self.dls['val']: 
            x, y = x.to(self.arg.device), y.to(self.arg.device)

            x_hat, vq_out = self.model(x)
            e_counter.update(vq_out['q'].unsqueeze(1).unsqueeze(1))

            x_np = x.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()

            for i in range(x_np.shape[0]):
                for m, fn in METRIC_FUNCS.items():
                    logs[m].append(fn(x_np[i], x_hat_np[i]))
            break

        plot_recons(x_np, x_hat_np, tag, self.arg.save_path)

        for m, vals in logs.items():
            logs[m] = sum(vals) / len(vals)
        util_ratio, _ = e_counter.compute_utilization()
        logs['util_ratio'] = util_ratio*100
        print(' | '.join([f'{k}: {v:.5f}' for k, v in logs.items()]), file=f)
        
        if wandb.run is not None:
            wandb.log(logs)