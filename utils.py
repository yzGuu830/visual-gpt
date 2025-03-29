import argparse
import os
import yaml
import torch
import json
import numpy as np
import random

import torchvision
import matplotlib.pyplot as plt


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


def read_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f) # dict
    return config


def img_grid_show(data, disp_num=256, fig_size=(10,20), show=False, save=None):
    """
    data: (B, C, H, W)
    """
    img = torchvision.utils.make_grid(data, nrow=int(np.sqrt(disp_num)), normalize=True)
    npimg = img.numpy()    
    plt.figure(figsize=fig_size)
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    if show: plt.show()