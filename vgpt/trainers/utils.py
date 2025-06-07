import os
import argparse
import torch
import matplotlib.pyplot as plt 


def plot_reconstructions(raws: list, recons: list, tag: str, save_path: str = None, max_shown: int = 16):
    """Plot raw and reconstructed images.
    Args:
        raws (list[np.ndarray]): list of reconstructed images, each numpy array with shape (C, H, W)
        recons (list[np.ndarray]): list of reconstructed images, each numpy array with shape (C, H, W)
        tag (str): title (filename) for the plot
        save_path (str): path to save the plot under
        max_shown (int): maximum number of samples to plot
    """
    raws = raws[:max_shown]
    recons = recons[:max_shown]

    num_samples = len(raws)
    assert num_samples % 2 == 0, "# of images must be divisible by 2, but found %d" % num_samples
    
    half_size = num_samples // 2
    fig, axes = plt.subplots(4, half_size, figsize=(half_size * 2, 10))

    for i in range(half_size):
        axes[0, i].imshow(raws[i].transpose(1, 2, 0))
        axes[0, i].axis('off')

        axes[2, i].imshow(raws[half_size+i].transpose(1, 2, 0))
        axes[2, i].axis('off')

    for i in range(half_size):
        axes[1, i].imshow(img_normalize(recons[i]).transpose(1, 2, 0))
        axes[1, i].axis('off')

        axes[3, i].imshow(img_normalize(recons[half_size+i]).transpose(1, 2, 0))
        axes[3, i].axis('off')

    plt.suptitle(tag)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'{tag}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()



def plot_generations(imgs: dict, tag: str, save_path: str = None):
    num_classes, num_samples = len(imgs), imgs.get('class1').size(0)

    fig, axes = plt.subplots(num_samples, num_classes, figsize=(num_classes*1.5, num_samples*2))
    for j, (k, v) in enumerate(imgs.items()):
        if num_samples == 1:
            axes[j].set_title(k, fontsize=10)
        else:
            axes[0, j].set_title(k, fontsize=10)
        
        for i in range(num_samples):
            img = img_normalize(v[i].cpu().numpy()).transpose(1, 2, 0)
            if num_samples == 1:
                axes[j].imshow(img)
                axes[j].axis('off')
            else:
                axes[i, j].imshow(img)
                axes[i, j].axis('off')

    plt.suptitle(tag)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'{tag}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()


def img_normalize(img, eps=1e-6):
    return (img - img.min()) / (img.max() - img.min() + eps)


def namespace2dict(config):
    if isinstance(config, argparse.Namespace):
        return {k: namespace2dict(v) for k, v in vars(config).items()}
    elif isinstance(config, list):
        return [namespace2dict(item) for item in config]
    else:
        return config

def save_checkpoint(model, save_path, save_name='model.pth'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


def load_checkpoint(model, load_path):
    model.load_state_dict(torch.load(load_path), map_location='cpu', strict=False, weights_only=True)