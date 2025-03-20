import argparse, torch
import wandb

from utils import *
from recon.trainer import Trainer as ReconTrainer



def parse_args_confs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--save_path', type=str, default='../results/default_exp')
    parser.add_argument('--conf_path', type=str, default='conf/base.yaml')
    parser.add_argument('--wandb_project', type=str, default=None)
    

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    confs = dict2namespace(read_yaml(args.conf_path))
    return args, confs


if __name__ == "__main__":
    
    arg, conf = parse_args_confs()
    trainer = ReconTrainer(arg, conf)

    if arg.wandb_project is not None:
        wandb.login()
        exp_name = arg.save_path.split('/')[-1]
        wandb.init(project=arg.wandb_project, name=exp_name)
    else:
        print("wandb disabled")

    trainer.train()
    wandb.finish()


"""
python train_tokenizer.py \
    --data_name cifar10 \
    --save_path ../outputs/resnetvq-cifar10 \
    --conf_path conf/recon/base.yaml \
    --wandb_project deepvq

"""