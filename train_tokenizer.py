import argparse, torch
import wandb

from utils import *
from recon.trainer import Trainer as ReconTrainer
from recon.trainer_adv import TrainerAdv as ReconTrainerAdv


def parse_args_confs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='../outputs/default_exp', help='path to save the outputs')
    parser.add_argument('--conf_path', type=str, default='conf/recon/base.yaml', help='path to the config file')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None, help='path to a pretrained tokenizer')
    parser.add_argument('--adv_training', action='store_true', help='whether to use adversarial training in VQ-GAN')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')
    parser.add_argument('--seed', type=int, default=53)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    confs = dict2namespace(read_yaml(args.conf_path))
    return args, confs


if __name__ == "__main__":
    
    arg, conf = parse_args_confs()
    seed_everything(arg.seed)
    
    if arg.adv_training:
        trainer = ReconTrainerAdv(arg, conf)
    else:
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
    --save_path ../outputs/resnetvq-cifar10 \
    --conf_path conf/recon/base.yaml \
    --wandb_project deepvq

python train_tokenizer.py \
    --save_path ../outputs/vqgan-imagenet100 \
    --conf_path conf/recon/vqgan_imagenet.yaml \
    --wandb_project deepvq

python train_tokenizer.py \
    --save_path ../outputs/vqgan-imagenet100-adv \
    --conf_path conf/recon/vqgan_imagenet_adv.yaml \
    --adv_training \
    --wandb_project deepvq
"""