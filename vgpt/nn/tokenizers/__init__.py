from .models import VisualTokenizer
from .taming.modules import NLayerDiscriminator
from .taming.losses import PerceptualLoss, ReconLoss, AdversarialLoss, calculate_adaptive_weight


from .quantizers.scalar_quantize import FSQ
from .quantizers.vector_quantize import VectorQuantize