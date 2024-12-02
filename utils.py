import argparse
import os
import yaml
import torch
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def dict2namespace(config: dict):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config: argparse.Namespace):
    return vars(config)


def read_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f) # dict
    return config


def save_checkpoint(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '/model.pth')


def load_checkpoint(model, load_path):
    model.load_state_dict(torch.load(load_path), map_location='cpu')