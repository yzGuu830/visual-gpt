import torch

class Metric:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class MSEMetric(Metric):
    def __call__(self, x, x_hat):
        mse = torch.mean((x - x_hat) ** 2)
        return mse.item()


class PSNRMetric(Metric):
    def __call__(self, x, x_hat):
        mse = torch.mean((x - x_hat) ** 2)
        if mse == 0:
            return float('inf')  # PSNR is infinite if there is no error
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()


class SSIMMetric(Metric):
    def __init__(self, data_range=1., size_average=True):
        from pytorch_msssim import SSIM
        self.ssim_fn = SSIM(data_range=data_range, 
                            size_average=size_average)

    def __call__(self, x, x_hat):
        ssim_value = self.ssim_fn(x, x_hat)
        return ssim_value.item()


class LPIPSMetric(Metric, torch.nn.Module):
    """note: require (x, x_hat) to be normalized in range [-1, 1]"""
    def __init__(self, net='vgg', device='cpu'):
        super(LPIPSMetric, self).__init__()
        import lpips
        self.lpips_model = lpips.LPIPS(net=net)
        self.lpips_model.to(device)
        self.lpips_model.eval()
        
    def __call__(self, x, x_hat, normalize=True):
        with torch.no_grad():
            dist = self.lpips_model(x, x_hat, normalize=normalize)
        return dist.mean().item()
    


class rFIDMetric:
    def __call__(self, *args, **kwds):
        pass


class rISMetric:
    def __call__(self, *args, **kwds):
        pass