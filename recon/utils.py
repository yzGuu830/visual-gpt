import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


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


METRIC_FUNCS = {
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


def plot_recons(raws: list, 
                recons: list, 
                tag: str, 
                save_path: str = None):
    """Plot raw and reconstructed images.
    Args:
        raws (list): list of raw images, each numpy array with shape (C, H, W)
        recons (list): list of reconstructed images, each numpy array with shape (C, H, W)
        tag (str): tag for the plot
        save_path (str): path to save the plot
    """
    raws = raws[:16] # plot at most 16 samples
    recons = recons[:16] # plot at most 16 samples

    num_samples = len(raws)
    assert num_samples % 2 == 0, "Batch size must be divisible by 2"
    
    half_size = num_samples // 2  # To split into two groups
    fig, axes = plt.subplots(4, half_size, figsize=(half_size * 2, 10))

    for i in range(half_size):
        axes[0, i].imshow(np.clip(raws[i].transpose(1, 2, 0), 0.0, 1.0))
        axes[0, i].axis('off')

        axes[1, i].imshow(np.clip(raws[half_size + i].transpose(1, 2, 0), 0.0, 1.0))
        axes[1, i].axis('off')

    for i in range(half_size):
        axes[2, i].imshow(np.clip(recons[i].transpose(1, 2, 0), 0.0, 1.0))
        axes[2, i].axis('off')

        axes[3, i].imshow(np.clip(recons[half_size + i].transpose(1, 2, 0), 0.0, 1.0))
        axes[3, i].axis('off')

    plt.suptitle(tag)

    if save_path is not None:
        if not os.path.exists(save_path): 
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'{tag}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()