import torch
import torch.nn.functional as F

from ._base import _BaseVectorQuantizeLayer
from ._utils import *


class VectorQuantize(_BaseVectorQuantizeLayer):
    """
    A unified Vector Quantization module that supports various improved training/algorithmic approaches.

    Parameters:
        num_codewords (`int`):
            The dictionary size of the codebook.
        embedding_dim (`int`):
            The dimension of the code-word vectors.
        z_dim (`int`):
            The dimension of the latent vectors.
        cos_dist (`bool`, `optional`, defaults to `False`): 
            Whether to use l2 normalized distance or not (Łancucki et al., 2020).
        proj_dim (`int`, `optional`, defaults to `None`):
            The dimension of the low-dimensional projection space.
        random_proj (`bool`, `optional`, defaults to `False`):
            Whether to use fixed random projection matrix or not (Chui et al., 2022).
        replace_freq (`int`, `optional`, defaults to `0`):
            Frequency to replace least recently used codewords (Implementation copied from Huh et al. (2023)).
        penalty_weight (`float`, `optional`, defaults to `0.0`):
            Weight to encourage uniform distance distributions. 

    In addition, code-factorization (Yu et al., 2022) is supported within the autoencoder models. 
    To enable pre-training autoencoders without vq layers, use `freeze_dict_forward_hook` (see recon/trainer.py)

    Example:
        ```python
        >>> from vector_quantize import VectorQuantize
        >>> vq_layer = VectorQuantize(num_codewords=512, embedding_dim=256, z_dim=256)
        
        >>> # training forward pass
        >>> vq_out = vq_layer(z_e)

        >>> # tokenizing & decoding
        >>> vq_layer.eval()
        >>> code = vq_layer.quantize(z_e)
        >>> z_q = vq_layer.dequantize(code)
        ```
    """
    def __init__(self,
                 num_codewords: int,
                 embedding_dim: int,
                 z_dim: int, 
                 cos_dist: bool = False, # use normalized l2 distance
                 proj_dim: int = None, # low-dimensional search
                 random_proj: bool = False, # use random projection matrix
                 replace_freq: int = 0, # reactivate dead codewords
                 penalty_weight: float = 0.0, # penalize non-uniform dists
                 **kwargs
                 ):
        super().__init__(num_codewords, embedding_dim, **kwargs)

        self.z_dim = z_dim
        self.cos_dist = cos_dist

        self.proj_dim = proj_dim
        self.random_proj = random_proj
        self._init_projection_layers()

        self.penalty_weight = penalty_weight

        if replace_freq > 0:
            lru_replacement(self, rho=0.01, timeout=replace_freq)
            print("[VectorQuantize] Enabled LRU replacement with frequency", replace_freq)

    def forward(self, z_e):
        """
        Args:
            z_e (Tensor): latent with shape (bsz, hw, dim)

        Returns:
            -- dict --
            z_q (Tensor): quantized latent with shape (bsz, hw, dim)
            q (Tensor): discrete indices with shape (bsz, hw)
            cm_loss (Tensor): commitment loss (update encoder)
            cb_loss (Tensor): codebook loss (update codebook)
        """         
        dists = compute_dist(z_e, 
                             self.codebook, 
                             cos_dist=self.cos_dist,
                             z_proj_matrix=self.z_proj_matrix,
                             c_proj_matrix=self.c_proj_matrix)
            
        q = dists.min(dim=-1).indices
        z_q = F.embedding(q, self.codebook)

        if self.training:
            z_q = ste(z_e, z_q)
        
        cm_loss, cb_loss = self.compute_loss(z_e, z_q)

        if self.penalty_weight > 0: # encourage dists to be uniform
            penalty_loss = self.penalty_loss(dists)
            cm_loss += self.penalty_weight * penalty_loss
        
        return {
            'z_e': z_e,
            'z_q': z_q,
            'q': q,
            'cm_loss': cm_loss,
            'cb_loss': cb_loss,
        }


    def compute_loss(self, z_e, z_q):
        cm_loss = F.mse_loss(z_q.detach(), z_e, reduction="none").mean([1, 2])
        cb_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        return cm_loss, cb_loss
    

    def penalty_loss(self, dists):
        # dists (bsz, *, num_codewords)
        similarity = F.softmax(-dists, dim=-1)
        kl = - torch.sum(similarity * similarity.log(), dim=-1)  # [bsz, *]
        return kl.mean([1])


    @torch.no_grad()
    def quantize(self, z_e):
        dists = compute_dist(z_e, 
                             self.codebook, 
                             cos_dist=self.cos_dist,
                             z_proj_matrix=self.z_proj_matrix,
                             c_proj_matrix=self.c_proj_matrix)
        q = dists.min(dim=-1).indices
        return q
    

    @torch.no_grad()
    def dequantize(self, q):
        z_q = F.embedding(q, self.codebook)
        return z_q


    def _init_projection_layers(self, ):
        if self.proj_dim is None:
            self.z_proj_matrix = None
            self.c_proj_matrix = None
            return 
        
        if self.proj_dim != self.z_dim:
            print("[VectorQuantize] Enabled latent factorization into low dimensional space ", end="")
            z_proj_matrix = init_proj_matrix(self.random_proj, self.z_dim, self.proj_dim)
            if self.random_proj:
                self.register_buffer('z_proj_matrix', z_proj_matrix)
                print("with random projection matrix")
            else:
                self.z_proj_matrix = z_proj_matrix
                print("with learnable projection matrix")
        
        if self.proj_dim != self.embedding_dim:
            print("[VectorQuantize] Enabled codebook factorization into low dimensional space ", end="")
            c_proj_matrix = init_proj_matrix(self.random_proj, self.embedding_dim, self.proj_dim)
            if self.random_proj:
                self.register_buffer('c_proj_matrix', c_proj_matrix)
                print("with random projection matrix")
            else:
                self.c_proj_matrix = c_proj_matrix
                print("with learnable projection matrix")

def ste(z, z_q):
    return z + (z_q - z).detach()