import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import wandb

from io import BytesIO
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity



def plot_reconstructions(raws: list, recons: list, tag: str, save_path: str = None, max_shown: int = 16):
    """Plot raw and reconstructed images.
    Args:
        raws (list[np.ndarray]): list of reconstructed images, each numpy array with shape (C, H, W)
        recons (list[np.ndarray]): list of reconstructed images, each numpy array with shape (C, H, W)
        tag (str): title (filename) for the plot
        save_path (str): path to save the plot under
        max_shown (int): maximum number of samples to plot
    """
    raws = raws[:max_shown]
    recons = recons[:max_shown]

    num_samples = len(raws)
    assert num_samples % 2 == 0, "# of images must be divisible by 2, but found %d" % num_samples
    
    half_size = num_samples // 2
    fig, axes = plt.subplots(4, half_size, figsize=(half_size * 2, 10))

    for i in range(half_size):
        axes[0, i].imshow(raws[i].transpose(1, 2, 0))
        axes[0, i].axis('off')

        axes[2, i].imshow(raws[half_size+i].transpose(1, 2, 0))
        axes[2, i].axis('off')

    for i in range(half_size):
        axes[1, i].imshow(img_normalize(recons[i]).transpose(1, 2, 0))
        axes[1, i].axis('off')

        axes[3, i].imshow(img_normalize(recons[half_size+i]).transpose(1, 2, 0))
        axes[3, i].axis('off')

    plt.suptitle(tag)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'{tag}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()



def plot_generations(imgs: dict, tag: str, save_path: str = None):
    num_classes, num_samples = len(imgs), imgs.get('class1').size(0)

    fig, axes = plt.subplots(num_samples, num_classes, figsize=(num_classes*1.5, num_samples*2))
    for j, (k, v) in enumerate(imgs.items()):
        if num_samples == 1:
            axes[j].set_title(k, fontsize=10)
        else:
            axes[0, j].set_title(k, fontsize=10)
        
        for i in range(num_samples):
            img = img_normalize(v[i].cpu().numpy()).transpose(1, 2, 0)
            if num_samples == 1:
                axes[j].imshow(img)
                axes[j].axis('off')
            else:
                axes[i, j].imshow(img)
                axes[i, j].axis('off')

    plt.suptitle(tag)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'{tag}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()


def img_normalize(img, eps=1e-6):
    return (img - img.min()) / (img.max() - img.min() + eps)


def namespace2dict(config):
    if isinstance(config, argparse.Namespace):
        return {k: namespace2dict(v) for k, v in vars(config).items()}
    elif isinstance(config, list):
        return [namespace2dict(item) for item in config]
    else:
        return config

def save_checkpoint(model, save_path, save_name='model.pth'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


def load_checkpoint(model, load_path):
    model.load_state_dict(torch.load(load_path), map_location='cpu', strict=False, weights_only=True)
    

class PSNR: 
    def __init__(self):
        self.name = 'psnr'
    def __call__(self, x, y):
        """x, y: (C, H, W)"""
        return peak_signal_noise_ratio(x, y, data_range=1)

class SSIM:
    def __init__(self):
        self.name = 'ssim'
    def __call__(self, x, y):
        """x, y: (C, H, W)"""
        return structural_similarity(x, y, channel_axis=0, data_range=1)

METRICS = {
    'psnr': PSNR(),
    'ssim': SSIM(),
}


class VQCodebookCounter:
    """
    Counter maintaining codebook utilization rate on a held-out validation set
    
    -- Usage Example --

    counter = EntropyCounter(
                codebook_size = 16,
                num_streams = 1,
                num_groups = 1
                )
    
    counter.reset_stats(num_hidden_layers)
    
    for data in validation_set:
        codes = quantize(data)
        counter.update(codes)
    
    total, util_per_vq = counter.compute_utilization()

    """
    def __init__(self, 
                 codebook_size=16, 
                 num_streams=1, # for residual vq 
                 num_groups=1,  # for group vq
                 device="cuda"):

        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.device = device

        self.reset_stats(num_streams)

    def reset_stats(self, num_streams):
        self.codebook_counts = {
                f"layer_{S+1}_group_{G+1}": torch.zeros(self.codebook_size, device=self.device) \
                    for S in range(num_streams) for G in range(self.num_groups)
                } # counts codeword stats for each codebook
        
        self.total_counts = 0
        self.dist = None    # posterior distribution for each codebook
        self.entropy = None # entropy stats for each codebook

        self.max_entropy_per_book = np.log2(self.codebook_size)
        self.max_total_entropy = num_streams * self.num_groups * self.max_entropy_per_book
        self.num_streams = num_streams

    def update(self, codes):
        """ Update codebook counts and total counts from a batch of codes
        Args:
            codes: (B, num_streams/num_layers, group_size, *)
        """ 
        assert codes.size(1) == self.num_streams and codes.size(2) == self.num_groups, "code indices size not match"
        num_codes = codes.size(0) * codes.size(-1)
        self.total_counts += num_codes

        for s in range(self.num_streams):
            stream_s_code = codes[:, s]                      # (B, group_size, *)
            for g in range(self.num_groups):
                stream_s_group_g_code = stream_s_code[:,g]   # (B, *)
                one_hot = F.one_hot(stream_s_group_g_code, num_classes=self.codebook_size) # (B, *, codebook_size)
                self.codebook_counts[f"layer_{s+1}_group_{g+1}"] += one_hot.view(-1, self.codebook_size).sum(0) # (*, codebook_size)
        
    def _form_distribution(self):
        """After iterating over a held-out set, compute posterior distribution for each codebook"""
        assert self.total_counts > 0, "No data collected, please update on a specific dataset"
        self.dist = {}
        for k, _counts in self.codebook_counts.items():
            self.dist[k] = _counts / torch.tensor(self.total_counts, device=_counts.device)
    
    def _form_entropy(self):
        """After forming codebook posterior distributions, compute entropy for each distribution"""
        assert self.dist is not None, "Please compute posterior distribution first using self._form_distribution()"
        
        self.entropy = {}
        for k, dist in self.dist.items():
            self.entropy[k] = (-torch.sum(dist * torch.log2(dist+1e-10))).item()
            
    def compute_utilization(self):
        """After forming entropy statistics for each codebook, compute utilization ratio (bitrate efficiency)
        Returns:
            -overall_utilization (float): overall utilization rate summed across all codebooks
            -utilization (dict): utilization rate for each codebook 
        """
        if self.dist is None: self._form_distribution()
        if self.entropy is None: self._form_entropy()
        
        utilization = {}
        for k, e in self.entropy.items():
            utilization[k] = round(e/self.max_entropy_per_book, 4)

        return round(sum(self.entropy.values())/self.max_total_entropy, 4), utilization


class LatentVisualizer:
    def __init__(self, save_dir='../output/baseline', use_tsne=False, max_points=2048, use_wandb=False):
        self.save_dir = os.path.join(save_dir, 'latent-vis')
        self.use_tsne = use_tsne
        self.max_points = max_points
        self.use_wandb = use_wandb
        if not self.use_wandb:
            os.makedirs(self.save_dir, exist_ok=True)

    def _reduce_dim(self, data):
        if self.use_tsne:
            reducer = TSNE(n_components=2, perplexity=30, max_iter=300)
        else:
            reducer = PCA(n_components=2)
        return reducer.fit_transform(data)

    def plot(self, z, z_q, codebook, q=None, step=0):
        """
        z: Tensor of shape (B, *, D), encoder outputs
        z_q: Tensor of shape (B, *, D), quantized encoder outputs
        codebook: Tensor of shape (K, D), codebook embeddings
        q: Tensor of shape (B, *) with code indices per encoder output
        step: Training step (used for saving file)
        """
        
        z = z.flatten(end_dim=-2)
        z_q = z_q.flatten(end_dim=-2)
        q = q.flatten(end_dim=-1) if q is not None else None
        
        num_sample = z.shape[0]
        if num_sample > self.max_points:
            idx = torch.randperm(num_sample)[:self.max_points]
            z = z[idx]
            z_q = z_q[idx]
            if q is not None:
                q = q[idx]

        z_low = self._reduce_dim(z.detach().cpu().numpy())
        z_q_low = self._reduce_dim(z_q.detach().cpu().numpy())
        if codebook is not None:
            codebook_low = self._reduce_dim(codebook.detach().cpu().numpy())


        plt.figure(figsize=(12, 8))

        plt.scatter(z_low[:, 0], z_low[:, 1], c='black', alpha=0.4, label='Encoder Output')
        plt.scatter(z_q_low[:, 0], z_q_low[:, 1], c='blue', alpha=0.4, 
                    edgecolors='red', linewidths=0.6, label='Quantized Encoder Output')
        if codebook is not None:
            plt.scatter(codebook_low[:, 0], codebook_low[:, 1], c='blue', alpha=0.4, label='Codebook')

        active_fraction = q.unique().numel() / codebook.shape[0] if q is not None else None

        plt.legend()
        if not self.use_wandb:
            plt.title(f'Latent Visualization @ Step {step} | Active Fraction: {active_fraction}')
            plt.tight_layout()
            path = os.path.join(self.save_dir, f'step_{step}.png')
            plt.savefig(path)
        else:
            fig = plt.gcf()
            fig_np = figure_to_numpy(fig)
            wandb.log({f"latent_viz": wandb.Image(fig_np, 
                                                  caption=f'Latent Visualization @ Step {step} | Active Fraction: {active_fraction}')
                                                  }, step=step)

        plt.close()


def figure_to_numpy(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    return img_array