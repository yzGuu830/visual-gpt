import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import matplotlib.cm as cm
import warnings

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
        if data.shape[1] <= 2:
            return data
        if self.use_tsne:
            return safe_reduce(TSNE, data, perplexity=30.0)
        else:
            return safe_reduce(PCA, data)

    def plot(self, z, z_q, codebook, q=None, step=0, apply_colormap=True):
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

        if codebook is None:
            z_low = self._reduce_dim(z.detach().cpu().numpy())
        else:
            z_and_c = torch.cat([z.detach().cpu(), codebook.detach().cpu()], dim=0) 
            z_and_c_low = self._reduce_dim(z_and_c.numpy())
            z_low = z_and_c_low[:z.shape[0]]
            codebook_low = z_and_c_low[z.shape[0]:]

        plt.figure(figsize=(12, 8))
        if q is not None:
            if apply_colormap: # use colormap for each code-word (used when K is small)
                if codebook.shape[0] > 20:
                    warnings.warn("Using colormap with more than 20 codewords may not display correctly. Consider setting `apply_colormap=False`.")
                colormap = cm.get_cmap('tab20', codebook.shape[0])
                q_np = q.detach().cpu().numpy()
                q_unique = np.unique(q_np)
                for idx in range(codebook.shape[0]):
                    color = colormap(idx % 20)
                    if idx in q_unique:
                        mask = (q_np == idx)
                        plt.scatter(
                            z_low[mask, 0], z_low[mask, 1],
                            color=color,
                            marker='o',
                            s=3,
                            alpha=0.5,
                        )
                    if codebook is not None:
                        if idx not in q_unique:
                            plt.scatter(
                            codebook_low[idx, 0], codebook_low[idx, 1],
                            facecolor='blue',
                            marker='x',
                            s=15,
                            alpha=0.8,
                            label=f'unused code-word idx.{idx+1}' if idx < 20 else None
                        )
                        else:
                            plt.scatter(
                                codebook_low[idx, 0], codebook_low[idx, 1],
                                facecolor=color,
                                marker='*',
                                edgecolor='red',
                                linewidths=0.25,
                                s=75,
                                alpha=0.9,
                                label=f'used codeword idx.{idx+1}' if idx < 20 else None
                            )
                plt.legend(
                    loc='center left',          
                    bbox_to_anchor=(1.02, 0.5),    
                    borderaxespad=0.,
                    fontsize=8
                )
                plt.tight_layout(rect=[0, 0, 0.85, 1])

            else: # avoid colormap (used when K is large)
                plt.scatter(z_low[:, 0], z_low[:, 1], alpha=0.5, s=3, marker='.', c='gray', label='latents (P_z)')
                if codebook is not None:
                    q_unique = q.unique().numpy()
                    C_z = codebook_low
                    Q_z = C_z[q_unique]
                    all_idx = np.arange(C_z.shape[0])
                    C_z_minus_Q_z = C_z[~np.isin(all_idx, q_unique)]
                    plt.scatter(Q_z[:, 0], Q_z[:, 1], alpha=0.9, label='used code-words (Q_z)', marker='*', s=50, c='blue', edgecolors='red', linewidths=0.25)
                    
                    if C_z_minus_Q_z.shape[0] > 0:
                        plt.scatter(C_z_minus_Q_z[:, 0], C_z_minus_Q_z[:, 1], 
                                    facecolor='blue',
                                    marker='x',
                                    s=25,
                                    alpha=0.8, label='unused code-words (C_z - Q_z)')
                plt.legend()

        else:
            plt.scatter(z_low[:, 0], z_low[:, 1], alpha=0.4, label='Encoder output z (P_z)', marker='.')
            if codebook is not None:
                plt.scatter(codebook_low[:, 0], codebook_low[:, 1], label='Codebook (C)', marker='*')
            plt.legend()       


        active_fraction = q.unique().numel() / codebook.shape[0] if q is not None else None
        if not self.use_wandb:
            plt.title(f'Latent Visualization @ Step {step} | Active Fraction: {active_fraction}')
            plt.tight_layout()
            path = os.path.join(self.save_dir, f'step_{step}.png')
            plt.savefig(path, dpi=200)
        else:
            fig = plt.gcf()
            fig_np = figure_to_numpy(fig)
            wandb.log({f"latent_viz": wandb.Image(fig_np, 
                                                  caption=f'Latent Visualization @ Step {step} | Active Fraction: {active_fraction}')
                                                  }, step=step)

        plt.close()


def safe_reduce(method, data, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return method(n_components=2, **kwargs).fit_transform(data)


def figure_to_numpy(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    return img_array