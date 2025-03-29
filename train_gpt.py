import argparse
import os
import torch
import wandb

from vgpt import GenTrainer
from utils import *

def parse_args_confs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='baseline', help='wandb experiment name')
    parser.add_argument('--output_path', type=str, default='../outputs', help='output path')
    
    parser.add_argument('--tokenizer_path', type=str, default='../outputs/vqgan-stfdogs', help='path to pretrained visual tokenizer')
    parser.add_argument('--conf', type=str, default='conf/stfdogs.yaml', help='path to the config file')
    
    parser.add_argument('--wandb', type=str, default=None, help='wandb project name')
    parser.add_argument('--seed', type=int, default=53)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.save_path = os.path.join(args.output_path, args.exp_name)

    confs = dict2namespace(read_yaml(args.conf))
    return args, confs

if __name__ == "__main__":
    arg, conf = parse_args_confs()
    seed_everything(arg.seed)

    if arg.wandb is not None:
        wandb.login()
        wandb.init(project=arg.wandb, name=arg.exp_name+"-recon")
    else:
        print("wandb disabled")

    trainer = GenTrainer(conf, arg)
    trainer.train(arg.tokenizer_path)
    wandb.finish()


"""
python train_gpt.py --exp_name vqgan-stfdogs  --output_path ../outputs  --tokenizer_path ../outputs/vqgan-stfdogs  --conf conf/stfdogs.yaml  --wandb visual-gpt


"""