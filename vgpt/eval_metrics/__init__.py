from .latent import VectorQuantEval, VectorQuantLatentVisualizer
from .vision import *

METRIC_FUNCS = {
    'MSE': MSEMetric,
    'PSNR': PSNRMetric,
    'SSIM': SSIMMetric,
    'LPIPS': LPIPSMetric,
    'rFID': rFIDMetric,
    'rIS': rISMetric
}