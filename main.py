import argparse, torch

from utils import *
from recon.trainer import Trainer as ReconTrainer



def parse_args_confs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--save_path', type=str, default='../results/default_exp')
    parser.add_argument('--conf_path', type=str, default='conf/base.yaml')
    

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    confs = dict2namespace(read_yaml(args.conf_path))
    return args, confs



if __name__ == "__main__":
    
    arg, conf = parse_args_confs()
    trainer = ReconTrainer(arg, conf)
    trainer.train()


"""
python main.py \
    --data_name cifar10 \
    --save_path ../results/default_exp \
    --conf_path conf/base.yaml

"""