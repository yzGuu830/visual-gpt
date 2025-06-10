import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb


from io import BytesIO
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class VectorQuantEval:
    """
    Evaluate codebook utilization on a held-out validation set
    
    -- Usage Example --

    vq_eval = VectorQuantEval(codebook_size = 256)
    
    vq_eval.reset_stats()
    for data in validation_set:
        codes = quantize(data)
        vq_eval.update(codes)
    
    eval_stats = vq_eval.cpt_stats()

    """
    def __init__(self, codebook_size=256):

        self.codebook_size = codebook_size
        self.max_entropy = np.log2(self.codebook_size)

    def reset_stats(self):
        self.counts = torch.zeros(self.codebook_size, dtype=torch.float64)
        self.total_counts = 0

    def update(self, codes):
        """ Batch update codebook counts and total counts
        Args:
            codes: (bsz, *)
        """ 
        self.total_counts += codes.numel()
        one_hot = torch.nn.functional.one_hot(codes, num_classes=self.codebook_size) # (bsz, *, codebook_size)
        self.counts += one_hot.view(-1, self.codebook_size).sum(0).detach().cpu()

    def cpt_stats(self):
        
        dist = self.counts / self.total_counts
        assert abs(dist.sum().item() - 1.0) < 1e-6
        
        entropy = (-torch.sum(dist * torch.log2(dist+1e-10))).item()
        ppl = 2 ** entropy
        active_rate = 100 * (self.counts > 0.0).float().mean().item()
        util_efficiency = 100 * (entropy / self.max_entropy)

        return {
            "entropy": entropy,
            "ppl": ppl,
            "active_rate": active_rate,
            "util_efficiency": util_efficiency
        }
    

class VectorQuantLatentVisualizer:
    """ Visualize latent space of vector quantization models """
    def __init__(self, 
                 save_dir='../output/baseline', 
                 use_tsne=False, 
                 max_points=2048, 
                 use_wandb=False):
        
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