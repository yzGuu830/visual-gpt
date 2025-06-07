import torch
import torch.nn as nn

import warnings

from ._utils import init_dict_forward_hook, freeze_dict_forward_hook

class _BaseVectorQuantizeLayer(nn.Module):
    INIT_METHODS = ['kaiming', 'latent_random', 'fixed']
    def __init__(self,
                 num_codewords: int,
                 embedding_dim: int,
                 pretrain_steps: int = 0,
                 init_method: str = 'kaiming',
                 ):
        super().__init__()
        
        self.num_codewords = num_codewords
        self.embedding_dim = embedding_dim

        self.pretrain_steps = pretrain_steps
        self.init_method = init_method

        if init_method not in self.INIT_METHODS:
            raise ValueError(f"Invalid initialization method '{init_method}'. "
                             f"Supported methods are: {self.INIT_METHODS}.")

        if init_method == 'latent_random' and pretrain_steps == 0:
            warnings.warn(
                "[VectorQuantize] Using 'latent_random' initialization for codebook without pretraining steps may lead to poor convergence. "
                "Consider setting `pretrain_steps` > 0 to stabilize the input latent vector first."
            )

        self._init_codebook()

        if pretrain_steps > 0:
            print(f"[VectorQuantize] autoencoder pretraining steps set to {pretrain_steps}")
            self.register_forward_hook(freeze_dict_forward_hook)
            self.register_buffer('done_steps', torch.tensor(0))

    def _init_codebook(self,):
        
        embed = torch.ones(self.num_codewords, self.embedding_dim)
        if self.init_method == 'fixed':
            embed.weight.data = hypersphere_codebook(self.num_codewords, self.embedding_dim)
            self.register_buffer('codebook', embed)
        
        elif self.init_method == 'kaiming':
            nn.init.kaiming_normal_(embed)
            self.codebook = nn.Parameter(embed, requires_grad=True)

        elif self.init_method == 'latent_random':
            self.codebook = nn.Parameter(embed, requires_grad=True)
            print("[VectorQuantize] codebook to be initialized from latents z_e")
            self.register_forward_hook(init_dict_forward_hook) # must be registered before freeze hook
            self.register_buffer('initialized', torch.tensor(0.)) 

    def prepare_inputs(self, z_e):
        """
        Reshape 2D Latent for Vector Quantization
        Args:
            z_e (Tensor): latent with shape (bsz, dim, h, w)
        Returns:
            z_e (Tensor): flattened latent with shape (bsz, hw, dim)
        """
        if z_e.dim() == 3:
            raise ValueError("Input tensor should have shape (bsz, dim, h, w), but got {}".format(z_e.shape))
        
        return z_e.flatten(2).transpose(1, 2)
    
    def recover_original(self, z_q, z_shape):
        """
        Recover Original Shape of 2D Latent
        Args:
            z_q (Tensor): quantized latent with shape (bsz, hw, dim)
            z_shape (tuple): original shape of latent (h, w)
        """
        bsz, _, dim = z_q.shape
        
        return z_q.transpose(1, 2).view(bsz, dim, *z_shape)

    def quantize(self, z_e):

        raise NotImplementedError
    
    def dequantize(self, q):

        raise NotImplementedError
    

def hypersphere_codebook(N, d):
    X = torch.randn(N, d)
    X = X / X.norm(dim=1, keepdim=True)
    return X