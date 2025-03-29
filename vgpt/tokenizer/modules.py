"""
Adapted convolution autoencoder modules following the original VQ-VAE paper https://arxiv.org/abs/1711.00937
"""
import torch.nn as nn



ACT_MAP = {"relu": nn.ReLU(), "elu": nn.ELU(1.0), 'leaky_relu': nn.LeakyReLU(0.01)}


NORM_MAP = {
    "bn": lambda channel: nn.BatchNorm2d(channel, eps=1e-6, affine=True), 
    "gn": lambda channel: nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True)
    }


class ResBlock(nn.Module):
    def __init__(self, in_dim, res_dim, act="relu", norm="bn"):
        super().__init__()
        self.block = nn.Sequential(
                ACT_MAP[act],
                nn.Conv2d(in_dim, res_dim, 3, 1, 1),
                NORM_MAP[norm](res_dim),
                ACT_MAP[act],
                nn.Conv2d(res_dim, in_dim, 1, 1, 0),
                NORM_MAP[norm](in_dim)
            )
    def forward(self, x):
        return x + self.block(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_dim=3, z_dim=256, resolution_depth=2, residual_depth=2, act="relu", norm="bn"):
        super(VAEEncoder, self).__init__()
        
        down = nn.ModuleList()      
        for i in range(resolution_depth-1, 0, -1):
            # downsample `resolution_depth - 1` times
            out_dim = z_dim // 2**i
            down.append(nn.Sequential(*[
                    nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1),
                    NORM_MAP[norm](out_dim),
                    ACT_MAP[act],
                ]))
            in_dim = out_dim
        down.append(nn.Conv2d(out_dim, z_dim, 4, stride=2, padding=1))
        self.down = down
        
        resblocks = [ ResBlock(z_dim, z_dim, act, norm) for _ in range(residual_depth) ]
        self.res_blk = nn.Sequential(*resblocks)

    def forward(self, x):
        for d_blk in self.down:
            x = d_blk(x)
        return self.res_blk(x)


class VAEDecoder(nn.Module):
    def __init__(self, in_dim=3, z_dim=256, resolution_depth=2, residual_depth=2, act="relu", norm="bn"):
        super(VAEDecoder, self).__init__()
        
        resblocks = [ ResBlock(z_dim, z_dim, act, norm) for _ in range(residual_depth) ]
        self.res_blk = nn.Sequential(*resblocks)
        
        up = nn.ModuleList()  
        for _ in range(1, resolution_depth):
            out_dim = dim_z // 2
            up.append(nn.Sequential(*[
                    nn.ConvTranspose2d(dim_z, out_dim, 4, stride=2, padding=1),
                    NORM_MAP[norm](out_dim),
                    ACT_MAP[act],
                ]))
            dim_z = out_dim
        up.append(nn.ConvTranspose2d(out_dim, in_dim, 4, stride=2, padding=1))
        self.up = up

    def forward(self, z):
        res_out = self.res_blk(z)
        for u_blk in self.up:
            res_out = u_blk(res_out)
        return res_out