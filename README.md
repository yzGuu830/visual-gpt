# DeepVQ

PyTorch Implementation of VQ-VAE based Auto-Regressive Visual Language Models


## Usage
### Environment Setup
```bash
conda create -n vlm python=3.9
pip install -r requirements.txt
```


### Training a Visual Tokenizer
```ruby
python train_tokenizer.py  --data_name cifar10  --save_path ../outputs/resnetvq-cifar10 --conf_path conf/recon/base.yaml --wandb_project deepvq
```
Following the description in the original [VQ-VAE paper](https://arxiv.org/abs/1711.00937), the visual tokenizer is composed of an autoencoder similar to those in a VAE. The implemented encoder-decoder here follows a simple symmetric and convolutional ResNet architecture. 

The vector quantization layer supports several improved training techniques. Below is an example programmatic usage of it:
```python
from vector_quantize import VectorQuantize

vq_layer = VectorQuantize(
        num_codewords = 1024, # dictionary size K
        embedding_dim = 256, # codeword dim d
        cos_dist = False, # use cosine-distance
        proj_dim = None, # low-dimensional factorization
        random_proj = False, # random projection search
)

# tokenizer & decoding
code = vq_layer.quantize(z_e)
z_q = vq_layer.dequantize(code)
```

During training, we use `wandb` for logging statistics. All model configuratinos and experimental parameters can be found under the `conf/recon` folder. 


### Training a Visual Language Model
``` ruby
python train_lm.py  --data cifar10  --from_pretrained ../outputs/resnetvq-cifar10  --output_path ../outputs  --conf_path conf/generative/base.yaml  --exp_name gpt2-cifar10  --lr 5e-6  --scheduler cosine  --wandb_project deepvq
```

Instead of using PixelCNN, we train GPT-2 language model on top of visual tokens produced from the VQ-VAE tokenizer. Specifically, this will train a class-conditional image generator based on standard transformer auto-regressive decoding. All model configuratinos and experimental parameters can be found under the `conf/generative` folder. 

Below is an example programmatic usage of a conditional transformer: 
```python
from synthesis.cond_transformer import CondTransformer
from tokenizer import VisualTokenizer

tokenizer = VisualTokenizer.from_pretrained("resnetvq-cifar10")
model = CondTransformer(
    visual_tokenizer=tokenizer,
    condition_vocab=10, # cifar10 classes
    max_pos_len=1024,
    d_model=768,
    num_transformer_layers=12,
    num_attn_heads=12,
)

# generate images
cond = torch.tensor([[1]]) # class 1
gen_imgs, _ = model.sample(cond, latent_size=(8,8), do_sample=True, temperature=0.8)
```

## Results