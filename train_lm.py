import argparse
import os
import wandb

from utils import *
from synthesis.trainer import Trainer

def parse_args_confs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, default='../outputs/resnetvq-cifar10', help='path to pretrained visual tokenizer')
    parser.add_argument('--output_path', type=str, default='../outputs', help='output path')
    parser.add_argument('--exp_name', type=str, default='baseline', help='wandb experiment name')
    parser.add_argument('--conf_path', type=str, default='conf/generative/base.yaml', help='path to config file')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')
    parser.add_argument('--seed', type=int, default=53)

    args = parser.parse_args()    
    args.save_path = os.path.join(args.output_path, args.exp_name)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    confs = dict2namespace(read_yaml(args.conf_path))
    return args, confs

if __name__ == "__main__":
    args, confs = parse_args_confs()
    seed_everything(args.seed)

    if args.wandb_project is not None:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.exp_name)
    else:
        print("wandb disabled")

    trainer = Trainer(args, confs)
    trainer.train()

"""
python train_lm.py \
    --tokenizer_path ../outputs/resnetvq-cifar10 \
    --output_path ../outputs \
    --conf_path conf/generative/base.yaml \
    --exp_name gpt2-resnetvq-cifar10 \
    --wandb_project deepvq

python train_lm.py \
    --tokenizer_path ../outputs/vqgan-imagenet100 \
    --output_path ../outputs \
    --conf_path conf/generative/gpt2_vqgan.yaml \
    --exp_name gpt2-vqgan-imagenet100 \
    --wandb_project deepvq
"""