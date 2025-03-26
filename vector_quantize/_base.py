import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseVectorQuantizeLayer(nn.Module):

    def __init__(self,
                 num_codewords: int,
                 embedding_dim: int,
                 learnable_codebook: bool = True,
                 ):
        super().__init__()
        
        self.num_codewords = num_codewords
        self.embedding_dim = embedding_dim
        self.learnable_codebook = learnable_codebook

        self._init_codebook()

    def _init_codebook(self,):

        embed = torch.ones(self.num_codewords, self.embedding_dim)
        nn.init.kaiming_normal_(embed)
        
        if self.learnable_codebook:
            self.codebook = nn.Parameter(embed, requires_grad=True)
        else:
            self.register_buffer('codebook', embed)

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