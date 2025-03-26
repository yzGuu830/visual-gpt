import torch.nn as nn 

from vector_quantize import VectorQuantize
from .modules import EncoderResnet, DecoderResnet
from .taming.modules import VQGANEncoder, VQGANDecoder


class VQGANModel(nn.Module):
    def __init__(self, 
                 ae_conf,
                 factorized_dim=None,
                 **vq_kwargs):
        super(VQGANModel, self).__init__()

        self.encoder = VQGANEncoder(**ae_conf)
        self.quantizer = VectorQuantize(**vq_kwargs)
        self.decoder = VQGANDecoder(**ae_conf)

        if factorized_dim is not None:
            self.in_proj = nn.Linear(ae_conf['z_channels'], factorized_dim, bias=False)
            self.out_proj = nn.Linear(factorized_dim, ae_conf['z_channels'], bias=False)
            print("factorized latents and codewords with dim = ", factorized_dim)

    def forward(self, x):
        z_e = self.encoder(x)

        b, c, h, w = z_e.shape
        z_e = z_e.flatten(2).transpose(1, 2)

        if hasattr(self, 'in_proj'):
            z_e = self.in_proj(z_e)
        
        vq_out = self.quantizer(z_e)
        z_q = vq_out['z_q']
        
        if hasattr(self, 'out_proj'):
            z_q = self.out_proj(z_q)

        z_q = z_q.transpose(1, 2).view(b, c, h, w)
        x_hat = self.decoder(z_q)
        return x_hat, vq_out


class VQResNet(nn.Module):
    def __init__(self,
                 ae_conf,
                 factorized_dim=None,
                 **vq_kwargs,
                 ):
        super(VQResNet, self).__init__()

        self.encoder = EncoderResnet(**ae_conf)
        self.quantizer = VectorQuantize(**vq_kwargs) 
        self.decoder = DecoderResnet(**ae_conf)

        if factorized_dim is not None:
            self.in_proj = nn.Linear(ae_conf['dim_z'], factorized_dim, bias=False)
            self.out_proj = nn.Linear(factorized_dim, ae_conf['dim_z'], bias=False)
            print("factorized latents and codewords with dim = ", factorized_dim)

    def forward(self, x):
        z_e = self.encoder(x)

        b, c, h, w = z_e.shape
        z_e = z_e.flatten(2).transpose(1, 2)

        if hasattr(self, 'in_proj'):
            z_e = self.in_proj(z_e)
        
        vq_out = self.quantizer(z_e)
        z_q = vq_out['z_q']
        
        if hasattr(self, 'out_proj'):
            z_q = self.out_proj(z_q)
        
        z_q = z_q.transpose(1, 2).view(b, c, h, w)
        x_hat = self.decoder(z_q)
        return x_hat, vq_out


def load_model(model_conf):    
    if model_conf.model_name == 'simple-ae':
        model = VQResNet(vars(model_conf.ae), model_conf.factorized_dim, **vars(model_conf.vq))

    elif model_conf.model_name == 'vqgan-ae':
        model = VQGANModel(vars(model_conf.ae), model_conf.factorized_dim, **vars(model_conf.vq))
    
    return model