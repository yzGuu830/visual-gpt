import functools
import torch.nn as nn 

from vector_quantize import VectorQuantize
from .modules import EncoderResnet, DecoderResnet, VQGANEncoder, VQGANDecoder


class VQGANModel(nn.Module):
    def __init__(self, 
                 ae_conf,
                 **vq_kwargs):
        super(VQGANModel, self).__init__()

        self.encoder = VQGANEncoder(**ae_conf)
        self.quantizer = VectorQuantize(**vq_kwargs)
        self.decoder = VQGANDecoder(**ae_conf)

    def forward(self, x):
        z_e = self.encoder(x)

        _, c, h, w = z_e.shape
        z_e = z_e.flatten(2).transpose(1, 2)
        
        vq_out = self.quantizer(z_e)
        
        z_q = vq_out['z_q'].transpose(1, 2).view(-1, c, h, w)
        x_hat = self.decoder(z_q)
        
        return x_hat, vq_out


class VQResNet(nn.Module):
    def __init__(self,
                 ae_conf,
                 **vq_kwargs,
                 ):
        super(VQResNet, self).__init__()

        self.encoder = EncoderResnet(**ae_conf)
        self.quantizer = VectorQuantize(**vq_kwargs) 
        self.decoder = DecoderResnet(**ae_conf)

    def forward(self, x):
        z_e = self.encoder(x)

        _, c, h, w = z_e.shape
        z_e = z_e.flatten(2).transpose(1, 2)
        
        vq_out = self.quantizer(z_e)
        
        z_q = vq_out['z_q'].transpose(1, 2).view(-1, c, h, w)
        x_hat = self.decoder(z_q)
        
        return x_hat, vq_out


def load_model(model_conf):    

    if model_conf.model_name == 'simple-ae':
        model = VQResNet(vars(model_conf.ae), **vars(model_conf.vq))

    elif model_conf.model_name == 'vqgan-ae':
        model = VQGANModel(vars(model_conf.ae), **vars(model_conf.vq))
    
    return model


class NLayerDiscriminator(nn.Module):
    """
    Copied from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)