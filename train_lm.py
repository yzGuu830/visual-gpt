import argparse
import os
import wandb

from utils import *
from synthesis.trainer import Trainer

def parse_args_confs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--from_pretrained', type=str, default='../outputs/resnetvq-coco17custom')
    parser.add_argument('--output_path', type=str, default='../outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_name', type=str, default='baseline', help='experiment name')
    parser.add_argument('--conf_path', type=str, default='conf/generative/base.yaml')
    parser.add_argument('--wandb_project', type=str, default=None)
    
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)

    args = parser.parse_args()    
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
    trainer.train(train_steps=args.train_steps, warmup_steps=args.warmup_steps, eval_every=args.eval_every, 
                  device=args.device, save_path=os.path.join(args.output_path, args.exp_name))

"""
python train_lm.py \
    --data cifar10 \
    --from_pretrained ../outputs/resnetvq-cifar10 \
    --output_path ../outputs \
    --conf_path conf/generative/base.yaml \
    --device cpu \
    --exp_name gpt2-cifar10 \
    --bsz 64 \
    --lr 5e-6 \
    --wd 1e-2 \
    --scheduler cosine \
    --seed 53 \
    --train_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 2000 \
    --wandb_project deepvq

python train_lm.py \
    --data coco2017custom \
    --from_pretrained ../outputs/resnetvq-coco2017custom \
    --output_path ../outputs \
    --conf_path conf/generative/base_coco.yaml \
    --device cpu \
    --exp_name gpt2-coco2017custom \
    --bsz 64 \
    --lr 5e-6 \
    --wd 1e-2 \
    --scheduler cosine \
    --seed 53 \
    --train_steps 10000 \
    --warmup_steps 1000 \
    --eval_every 2000 \
    --wandb_project deepvq

"""