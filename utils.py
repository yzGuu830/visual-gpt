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


CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def vis_gens(imgs: torch.Tensor, iteration, save_path: str = None):
    num_samples = imgs.size(0)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*1.75, 4))
    for i in range(num_samples):
        axes[i].imshow(imgs[i].cpu().numpy().transpose(1, 2, 0))
        axes[i].set_title(CIFAR10_LABELS[i])
        axes[i].axis('off')

    plt.suptitle(f'LM Generated Imgs @ Iteration {iteration}')
    
    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'iteration_{iteration}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()

def save_stats(loss_curve: list, save_path: str):
    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve)
    plt.xlabel('training step')
    plt.ylabel('nll loss')
    plt.grid()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'), bbox_inches='tight', dpi=120)
    return

def save_model(model, save_path: str):
    if not os.path.exists(save_path): os.makedirs(save_path)
    torch.save(model.lm.state_dict(), os.path.join(save_path, 'lm_checkpoint.pth'))
    json.dump(model.config, open(os.path.join(save_path, 'config.json'), 'w'))
    return