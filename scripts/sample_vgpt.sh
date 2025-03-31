#!/bin/bash

# Visual Transformer Autoregressive Decoding with Classifier-rejection

# Example usage:
# bash scripts/sample_vgpt.sh \
#   --from_pretrained ../outputs/vqgan-stfdogs \
#   --cls_name maltese samoyed australian_terrier \
#   --accept_n 20 \
#   --temperature 1.3 \
#   --top_k 100 \
#   --z_shape 16,16 \
#   --save_path stf_dogs_generated.gif

python sample.py "$@"