import torch, json
import torch.nn as nn

from recon.models import VQResNet, VQGANModel


class VisualTokenizer(nn.Module):
    def __init__(self,):
        super(VisualTokenizer, self).__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str="vq_resnet"):
        
        config = json.load(open(f"{pretrained_model_name_or_path}/config.json", "r"))
        if config['model_name'] == "vqgan-ae":
            model = VQGANModel(config['ae'], config['factorized_dim'], **config['vq'])
        elif config['model_name'] == "simple-ae":
            model = VQResNet(config['ae'], config['factorized_dim'], **config['vq'])
        model.load_state_dict(torch.load(f"{pretrained_model_name_or_path}/model.pth", map_location="cpu", weights_only=True), strict=False)
        print("visual tokenizer loaded from pretrained!")
        
        model.eval()
        tokenizer = cls()
        tokenizer.model = model

        return tokenizer
    
    @torch.no_grad()
    def encode(self, x):
        z_e = self.model.encoder(x)
        b, c, h, w = z_e.shape
        z_e = z_e.permute(0, 2, 3, 1).view(b, -1, c)
        code = self.model.quantizer.quantize(z_e)
        return code # (bsz, hw)
    
    @torch.no_grad()
    def decode(self, code):
        # code (bsz, h, w)
        z_q = self.model.quantizer.dequantize(code)
        z_q = z_q.permute(0, 3, 1, 2)
        x_hat = self.model.decoder(z_q)
        return x_hat


if __name__ == "__main__":
    model = VisualTokenizer.from_pretrained(
        "results/resnet_celeba_baseline"
        )
    x = torch.rand(2, 3, 64, 64)
    code = model.encode(x)
    print(f"Code: {code.shape}")
    x_hat = model.decode(code)
    print(f"Reconstructed Image: {x_hat.shape}")