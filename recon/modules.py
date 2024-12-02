import torch.nn as nn


ACT_FUNC = {"relu": nn.ReLU(), "elu": nn.ELU()}


class ResBlock(nn.Module):
    def __init__(self, dim, act="relu"):
        super().__init__()
        self.block = nn.Sequential(
                ACT_FUNC[act],
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.BatchNorm2d(dim),
                ACT_FUNC[act],
                nn.Conv2d(dim, dim, 1),
                nn.BatchNorm2d(dim)
            )

    def forward(self, x):
        return x + self.block(x)


class EncoderResnet(nn.Module):
    def __init__(self, in_dim=3, dim_z=64, num_scales=2, num_rb=2):
        super(EncoderResnet, self).__init__()
        
        layers_conv = nn.ModuleList()      
        for i in range(num_scales-1, 0, -1):
            out_dim = dim_z // 2**i
            layers_conv.append(nn.Sequential(*[
                    nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                ]))
            in_dim = out_dim
        layers_conv.append(nn.Conv2d(out_dim, dim_z, 4, stride=2, padding=1))
        self.conv = layers_conv
        
        layers_resblocks = [
                ResBlock(dim_z) for _ in range(num_rb)
            ]
        self.res = nn.Sequential(*layers_resblocks)

    def forward(self, x):
        for d_blk in self.conv:
            x = d_blk(x)
        z_e = self.res(x)
        return z_e

    def _verbose(self, x):
        print(f'input shape: {x.shape}')
        for i, d_blk in enumerate(self.conv):
            x = d_blk(x)
            print(f'[conv blk {i+1} out] feature shape: {x.shape}')
        z_e = self.res(x)
        return z_e


class DecoderResnet(nn.Module):
    def __init__(self, in_dim=3, dim_z=64, num_scales=2, num_rb=2):
        super(DecoderResnet, self).__init__()
        
        layers_resblocks = [
                ResBlock(dim_z) for _ in range(num_rb)
            ]
        self.res = nn.Sequential(*layers_resblocks)
        
        layers_convt = nn.ModuleList()  
        for i in range(1, num_scales):
            out_dim = dim_z // 2
            layers_convt.append(nn.Sequential(*[
                    nn.ConvTranspose2d(dim_z, out_dim, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(True),
                ]))
            dim_z = out_dim
        layers_convt.append(nn.ConvTranspose2d(out_dim, in_dim, 4, stride=2, padding=1))
        self.convt = layers_convt

    def forward(self, z):
        out_res = self.res(z)
        for u_blk in self.convt:
            out_res = u_blk(out_res)
        return nn.functional.sigmoid(out_res)

    def _verbose(self, z):
        out_res = self.res(z)
        print(f'latent shape: {out_res.shape}')
        for i, u_blk in enumerate(self.convt):
            out_res = u_blk(out_res)
            print(f'[convt blk {i+1} out] feature shape: {out_res.shape}')
        return out_res
    

if __name__ == "__main__":
    import torch
    x = torch.ones(1,3,32,32)

    enc = EncoderResnet(in_dim=3, dim_z=64, num_scales=2, num_rb=1)
    dec = DecoderResnet(in_dim=3, dim_z=64, num_scales=2, num_rb=1)

    z_e = enc._verbose(x)

    x_hat = dec._verbose(z_e)