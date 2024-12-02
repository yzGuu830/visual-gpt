import torch.nn as nn 

from vector_quantize import VectorQuantize
from .modules import EncoderResnet, DecoderResnet

class VQResNet(nn.Module):
    def __init__(self,
                 in_dim: int = 3, # input dim
                 dim_z: int = 64, # latent dim 
                 num_scales: int = 4, # number of resolution scales
                 blk_depth: int = 2,  # number of res blocks in each layer 
                 **vq_kwargs,
                 ):
        super(VQResNet, self).__init__()

        self.encoder = EncoderResnet(in_dim, dim_z, num_scales, blk_depth)
        self.quantizer = VectorQuantize(**vq_kwargs) 
        self.decoder = DecoderResnet(in_dim, dim_z, num_scales, blk_depth)

    def forward(self, x):
        z_e = self.encoder(x)

        _, c, h, w = z_e.shape
        z_e = z_e.flatten(2).transpose(1, 2)
        
        vq_out = self.quantizer(z_e)
        
        z_q = vq_out['z_q'].transpose(1, 2).view(-1, c, h, w)
        x_hat = self.decoder(z_q)
        
        return x_hat, vq_out


def load_model(conf):    

    vq_kwargs = {
                    'num_codewords': conf.num_codewords, 
                    'embedding_dim': conf.embedding_dim,
                    'cos_dist': conf.cos_dist,
                    'proj_dim': conf.proj_dim,
                    'random_proj': conf.random_proj,
                    'learnable_codebook': conf.learnable_codebook,
                }
    
    model = VQResNet(in_dim=conf.in_dim, 
                     dim_z=conf.dim_z, 
                     num_scales=conf.num_scales, 
                     blk_depth=conf.blk_depth, 
                     **vq_kwargs)
    
    return model 