import os
import json
import warnings
import torch
import torch.nn as nn 

from ..vector_quantize import VectorQuantize
from ..scalar_quantize import FSQ
from .modules import VAEEncoder, VAEDecoder
from .taming.modules import VQGANEncoder, VQGANDecoder



class VisualTokenizer(nn.Module):
    def __init__(self, ae_name: str, qtz_name: str, ae_conf: dict, **vq_kwargs):
        super(VisualTokenizer, self).__init__()

        self.init_modules(ae_name, qtz_name, ae_conf, vq_kwargs)

    @torch.no_grad()
    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.flatten2d(z_e)
        z_e = self.in_proj(z_e)
        code = self.quantizer.quantize(z_e)
        return code # (bsz, hw)
    
    @torch.no_grad()
    def decode(self, code):
        # code (bsz, h, w)
        z_q = self.quantizer.dequantize(code)
        z_q = self.out_proj(z_q)
        x_hat = self.decoder(z_q.permute(0, 3, 1, 2))
        return x_hat

    def forward(self, x):

        z_e = self.encoder(x)
        h, w = z_e.shape[-2:]
        z_e = self.flatten2d(z_e)

        z_e = self.in_proj(z_e)
        vq_out = self.quantizer(z_e)
        z_q = vq_out.get('z_q')
        z_q = self.out_proj(z_q)

        z_q = self.unflatten2d(z_q, h, w)
        x_hat = self.decoder(z_q)

        return x_hat, vq_out

    def init_modules(self, ae_name, qtz_name, ae_conf, vq_kwargs):
        ENC_MAP = {"vae": VAEEncoder, "vqgan": VQGANEncoder}
        DEC_MAP = {"vae": VAEDecoder, "vqgan": VQGANDecoder}
        QTZ_MAP = {"vq": VectorQuantize, "fsq": FSQ}

        if not isinstance(ae_conf, dict):
            ae_conf = vars(ae_conf)
        
        self.encoder = ENC_MAP[ae_name](**ae_conf)
        self.decoder = DEC_MAP[ae_name](**ae_conf)
        self.quantizer = QTZ_MAP[qtz_name](**vq_kwargs)

        z_dim = ae_conf.get('z_dim') or ae_conf.get('z_channels')
        c_dim = vq_kwargs.get('embedding_dim')
        
        proj = z_dim != c_dim
        self.in_proj = nn.Linear(z_dim, c_dim, bias=False) if proj else nn.Identity()
        self.out_proj = nn.Linear(c_dim, z_dim, bias=False) if proj else nn.Identity()
        if proj: print(f"[VectorQuantize] factorize latents into low dimension = {c_dim}")
        if z_dim < c_dim:
            warnings.warn(f"found z_dim={z_dim} and c_dim={c_dim}, codeword dimension might be set too large")


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str="vqgan-imagenet"):
        if not os.path.exists(pretrained_model_name_or_path):
            raise FileNotFoundError(f"Pretrained model {pretrained_model_name_or_path} not found!")
        
        conf = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json"), "r"))
        
        tokenizer = cls(conf['ae_name'], conf['qtz_name'], conf['ae_conf'], **conf['vq_conf'])

        tokenizer.load_state_dict(
            torch.load(os.path.join(pretrained_model_name_or_path, "model.bin"), map_location="cpu", weights_only=True), strict=False)
        print(f"visual tokenizer loaded from pretrained {pretrained_model_name_or_path}!")
        
        tokenizer.eval()
        return tokenizer
    
    def flatten2d(self, x):
        # x: (b, c, h, w) -> (b, h*w, c)
        return x.flatten(2).transpose(1, 2)
    
    def unflatten2d(self, x, h, w):
        # x: (b, h*w, c) -> (b, c, h, w)
        c = x.size(-1)
        return x.transpose(1, 2).view(-1, c, h, w)
    
    def get_last_layer(self):
        if hasattr(self.decoder, 'conv_out'): # VQGANDecoder
            return self.decoder.conv_out.weight
        
        if hasattr(self.decoder, 'up'): # VAEDecoder
            return self.decoder.up[-1].weight
    