import torch

class Metric:
    def __init__(self, 
                 data_range=[0, 1], 
                 in_channels=3):
        self.data_range = data_range
        self.in_channels = in_channels
        
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class MSEMetric(Metric):
    def __init__(self, data_range=[0, 1], in_channels=3):
        super(MSEMetric, self).__init__(data_range=data_range, in_channels=in_channels)

    def __call__(self, x, x_hat):
        mse = torch.mean((x - x_hat) ** 2)
        return mse.item()


class MAEMetric(Metric):
    def __init__(self, data_range=[0, 1], in_channels=3):
        super(MAEMetric, self).__init__(data_range=data_range, in_channels=in_channels)

    def __call__(self, x, x_hat):
        mae = torch.mean(torch.abs(x - x_hat))
        return mae.item()


class PSNRMetric(Metric):
    def __init__(self, data_range=[0, 1], in_channels=3):
        super(PSNRMetric, self).__init__(data_range=data_range, in_channels=in_channels)

    def __call__(self, x, x_hat):
        mse = torch.mean((x - x_hat) ** 2)
        psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))
        return psnr.item()


class SSIMMetric(Metric):
    def __init__(self, data_range=[0, 1], in_channels=3):
        super(SSIMMetric, self).__init__(data_range=data_range, in_channels=in_channels)

        from pytorch_msssim import SSIM
        self.ssim_fn = SSIM(data_range=data_range[1], channel=in_channels)

    def __call__(self, x, x_hat):
        ssim_value = self.ssim_fn(x, x_hat)
        return ssim_value.item()


class LPIPSMetric(Metric, torch.nn.Module):
    """note: require (x, x_hat) to be normalized in range [-1, 1]"""
    def __init__(self, data_range=[0, 1], in_channels=3, net='vgg', device='cpu'):
        torch.nn.Module.__init__(self)
        super(LPIPSMetric, self).__init__(data_range=data_range, in_channels=in_channels)

        if self.data_range[0] != 0 or self.data_range[1] != 1:
            Warning(f"LPIPSMetric expects data_range to be [0, 1], but got {self.data_range}. ")
        
        import lpips
        self.lpips_model = lpips.LPIPS(net=net)
        self.lpips_model.to(device)
        self.lpips_model.eval()

    @torch.no_grad()    
    def __call__(self, x, x_hat, normalize=True):
        """normalize: set True to adjust [0,1] input to [-1,1]"""
        dist = self.lpips_model(x, x_hat, normalize=normalize)
        return dist.mean().item()
    


class rFIDMetric:
    def __init__(self, data_range=[0, 1], in_channels=3):
        super(rFIDMetric, self).__init__(data_range=data_range, in_channels=in_channels)

    def __call__(self, *args, **kwds):
        pass


class rISMetric:
    def __init__(self, data_range=[0, 1], in_channels=3):
        super(rISMetric, self).__init__(data_range=data_range, in_channels=in_channels)

    def __call__(self, *args, **kwds):
        pass