import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dist(z_e, codebook, cos_dist=False, proj_matrix=None):
    """
    Args:
        z_e (Tensor): flattened latent with shape (bsz, *, embedding_dim), * denotes flattened quantized dimensions
        codebook (Tensor): embedding weight tensor with shape (num_codewords, embedding_dim)
        cos_dist (bool): whether to use cosine distance or not
        proj_matrix (Tensor): projection matrix with shape (embedding_dim, proj_dim) for low-dimensional search

    Returns:
        dists (Tensor): distance between z_e and each codewords in codebook with shape (bsz, *, num_codewords)
    """

    if cos_dist:
        z_e = F.normalize(z_e, p=2, dim=-1)
        codebook = F.normalize(codebook, p=2, dim=-1)

    if proj_matrix is not None:
        z_e = z_e @ proj_matrix
        codebook = codebook @ proj_matrix

    dists = torch.cdist(z_e, codebook, p=2) # (bsz, *, num_codewords)
    return dists


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
        nn.init.kaiming_normal_(embed) # TODO: need other initialization for not learnable codebook
        
        if self.learnable_codebook:
            self.codebook = nn.Parameter(embed, requires_grad=True)
        else:
            self.register_buffer('codebook', embed)

    def prepare_inputs(self, z_e):

        raise NotImplementedError
    
    def recover_original(self, z_q):

        raise NotImplementedError

    def quantize(self, z_e):

        raise NotImplementedError
    
    def dequantize(self, q):

        raise NotImplementedError