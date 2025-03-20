from synthesis.trainer import run

if __name__ == "__main__":
    run()

"""
python test.py \
    --data cifar10 \
    --from_pretrained outputs/resnet_cifar10_large \
    --output_path ./outputs \
    --num_classes 10 \
    --device cuda \
    --exp_name gpt2-baseline \
    --bsz 64 \
    --lr 5e-6 \
    --wd 1e-2 \
    --scheduler cosine \
    --seed 53 \
    --train_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 2000

"""