import torch
import torch.nn as nn
import torch.nn.functional as F


from ._base import _BaseVectorQuantizeLayer, compute_dist


def ste(z, z_q):
    return z + (z_q - z).detach()


class VectorQuantize(_BaseVectorQuantizeLayer):
    def __init__(self,
                 num_codewords: int,
                 embedding_dim: int,
                 cos_dist: bool = False,
                 proj_dim: int = None,
                 random_proj: bool = False,
                 **kwargs
                 ):
        super().__init__(num_codewords, embedding_dim, **kwargs)

        self.cos_dist = cos_dist

        self.proj_dim = proj_dim
        self.random_proj = random_proj
        self._init_projection_layers()

    def _init_projection_layers(self, ):
        if self.proj_dim is None: 
            self.proj_matrix = None
            return 
        
        print("[VectorQuantize] Enabled factorization into low dimensional space ", end="")
        if not self.random_proj:
            self.proj_matrix = nn.Parameter(
                torch.empty(self.embedding_dim, self.proj_dim), requires_grad=True)
            print("with learnable projection matrix")
        else:
            proj_matrix = torch.empty(self.embedding_dim, self.proj_dim)
            nn.init.xavier_normal_(proj_matrix) # TODO: need other distribution for random projection
            self.register_buffer('proj_matrix', proj_matrix)
            print("with random projection matrix")

    def forward(self, z_e):
        """
        Args:
            z_e (Tensor): latent with shape (bsz, *, embedding_dim), * denotes flattened quantized dimensions

        Returns:
            -- dict --
            z_q (Tensor): quantized latent with shape (bsz, *, embedding_dim)
            q (Tensor): discrete indices with shape (bsz, *)
            cm_loss (Tensor): commitment loss (update encoder)
            cb_loss (Tensor): codebook loss (update codebook)
        """
        
        with torch.no_grad():
            dists = compute_dist(z_e, 
                                 self.codebook, 
                                 cos_dist=self.cos_dist, 
                                 proj_matrix=self.proj_matrix)
            
        q = dists.min(dim=-1).indices
        z_q = F.embedding(q, self.codebook)

        if self.training:
            z_q = ste(z_e, z_q)

        cm_loss, cb_loss = self.compute_loss(z_e, z_q)

        return {
            'z_q': z_q,
            'q': q,
            'cm_loss': cm_loss,
            'cb_loss': cb_loss,
        }

    def compute_loss(self, z_e, z_q):
        cm_loss = F.mse_loss(z_q.detach(), z_e, reduction="none").mean([1, 2])
        cb_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        return cm_loss, cb_loss


    @torch.no_grad()
    def quantize(self, z_e):
        dists = compute_dist(z_e, 
                             self.codebook, 
                             cos_dist=self.cos_dist, 
                             proj_matrix=self.proj_matrix)
        q = dists.min(dim=-1).indices
        return q
    
    @torch.no_grad()
    def dequantize(self, q):
        z_q = F.embedding(q, self.codebook)
        return z_q

