# Visual-GPT

![example_generation](assets/stf_dogs_generated2.gif)

This repository provides a PyTorch implementation of Vector-Quantization (VQ)-based Autoregressive Vision Generative Models. Built upon the foundational VQ-GAN architecture, this repo integrates several improved techniques for training visual tokenizers, with a purpose of facilitating replication and research in autoregressive vision generative models.

[Results](#results) and [pretrained model checkpoints](#synthesize-your-dog-images) of class-conditioned image synthesizers on StanfordDogs, a dataset consisting of 120 dog breeds, are provided below. 


## Usage
### Setup
```bash
conda create -n vgpt python=3.9
cd visual-gpt
pip install -r requirements.txt
```

### Synthesize Your Dog Images!
To use the pretrained model, download directly from [this link](https://drive.google.com/drive/folders/1yU_cQifIvcWiesg18Lmf6EQIcprDBvfL?usp=sharing), and run:

```bash
bash scripts/sample_vgpt.sh \
    --from_pretrained path/to/vqgan-stfdogs \
    --cls_name maltese samoyed australian_terrier \ # dog classes
    --accept_n 10 \ # choose best 1/accept_n with classifier-rejection
    --temperature 1.3 --top_k 100 --top_p 0.9 # generation params.
```
This will generate a GIF visualizing the autoregressive decoding process, similar to the one shown at the top of this page.


### Training a Visual Tokenizer

The provided visual tokenizers are convolutional autoencoder-based. Implementations include a basic structure from the original VQ-VAE paper and a more advanced VQ-GAN architecture adapted from [taming transformers](https://github.com/CompVis/taming-transformers/tree/master). All configuration parameters are managed under `conf/exp.yaml`. 

To train a visual tokenizer:

```ruby
python train_tokenizer.py  --exp_name vqgan-stfdogs  --output_path ../outputs  --conf conf/stfdogs.yaml  --wandb visual-gpt
```

The implemented `VectorQuantization` layer supports several enhanced dictionary learning techniques. Example programmatic usage:

```python
from vgpt import VectorQuantize

vq_layer = VectorQuantize(
        num_codewords = 1024, # dictionary size K
        embedding_dim = 256, # codeword dim d
        cos_dist = False, # use cosine-distance
        proj_dim = None, # low-dimensional factorization
        random_proj = False, # random projection search
        penalty_weight = 0.25, # penalize non-uniform dists
        pretrain_steps = 5000, # warm-start autoencoders
        init_method = "latent_random" # initialize codebook with pre-trained latents
)
# tokenizing & decoding
code = vq_layer.quantize(z_e)
z_q = vq_layer.dequantize(code)
```


### Training a Visual GPT
After obtaining a fine-grained visual tokenizer, a visual language model can be easily trained for as many generative tasks as possible. To train a class-conditioned image synthesizer based on flattened image tokens:
``` ruby
python train_gpt.py --exp_name vqgan-stfdogs  --output_path ../outputs  --tokenizer_path ../outputs/vqgan-stfdogs  --conf conf/stfdogs.yaml  --wandb visual-gpt
```

The visual autoregressive model is implemented as `CondVisualGPT`, which is heavily relied on HuggingFace-Transformers🤗. It handles language model training and sampling very conveniently. 


## Results
Here are additional generated samples: 
![samples](assets/dog_breeds_samples.png)


## Acknowledgements
This repo is heavily inspired by these papers:
```bibtex
@misc{esser2021taming,
  title={Taming transformers for high-resolution image synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12873--12883},
  year={2021}
}
@misc{huh2023straightening,
  title={Straightening out the straight-through estimator: Overcoming optimization challenges in vector quantized networks},
  author={Huh, Minyoung and Cheung, Brian and Agrawal, Pulkit and Isola, Phillip},
  booktitle={International Conference on Machine Learning},
  pages={14096--14113},
  year={2023},
}
```