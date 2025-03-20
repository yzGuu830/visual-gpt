# from synthesis.cond_transformer import CondTransformer
# from synthesis.tokenizer import VisualTokenizer

# import torch
# from utils import img_grid_show


# tokenizer = VisualTokenizer.from_pretrained("outputs/resnet_cifar10_large")

# model = CondTransformer(
#     visual_tokenizer=tokenizer,
#     condition_vocab=10,
#     max_pos_len=1024,
#     d_model=64,
#     num_transformer_layers=4,
#     num_attn_heads=2,
# )

# x = torch.rand(2, 3, 32, 32)
# cond = torch.tensor([[1], 
#                         [2]])
# loss, lm_logits = model(x, cond)
# print(f"Loss: {loss}, Logits: {lm_logits.shape}")

# x_hat, code = model.sample(cond, latent_size=(8,8), do_sample=False)
# print(x_hat.shape)
# print(code.shape, code)

# from data import make_dl

# dls = make_dl('cifar10')

# X, y = next(iter(dls['train']))
# print(X.shape, y.shape)


# img_grid_show(X, disp_num=256, fig_size=(10,20), save="test.png")


# code = tokenizer.encode(X)
# print(code.shape)
# x_hat = tokenizer.decode(code.view(-1,8,8))
# print(x_hat.shape) # (bsz, 3, 32, 32) 32x32 image

# img_grid_show(x_hat, disp_num=256, fig_size=(10,20), save="test_recon.png") # (bsz, 3, 32, 32) 32x32 image

from synthesis.trainer import run


if __name__ == "__main__":
    run()





"""
python test.py \
        --data cifar10 \
        --from_pretrained outputs/resnet_cifar10_large \
        --output_path ./outputs \
        --num_classes 10 \
        --device cpu \
        --exp_name gpt2-baseline \
        --bsz 32 \
        --lr 3e-4 \
        --seed 42 \
        --train_steps 5000 \
        --eval_every 1000
"""