import torch
import os
import matplotlib.pyplot as plt


CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
COCO2017CUSTOM_LABELS = ['car', 'bird', 'cat', 'dog']
IMAGENET100_LABELS = ["Great Dane", "coyote", "red fox", "wild boar", "vizsla", "komondor", "Doberman", "hare", "boxer", "tabby", "gibbon", "African hunting dog"]
def vis_gens(imgs: torch.Tensor, tag: str, save_path: str = None):
    num_classes, num_samples = imgs.size(0), imgs.size(1)

    # needs refactoring
    if num_classes == len(CIFAR10_LABELS):
        labels = CIFAR10_LABELS
    elif num_classes == len(COCO2017CUSTOM_LABELS):
        labels = COCO2017CUSTOM_LABELS
    elif num_classes == len(IMAGENET100_LABELS):
        labels = IMAGENET100_LABELS

    fig, axes = plt.subplots(num_samples, num_classes, figsize=(num_classes, 10))
    for j in range(num_classes):
        if num_samples == 1:
            axes[j].set_title(labels[j])
        else:
            axes[0, j].set_title(labels[j])
        
        for i in range(num_samples):
            if num_samples == 1:
                axes[j].imshow(imgs[j, i].cpu().numpy().transpose(1, 2, 0))
                axes[j].axis('off')
            else:
                axes[i, j].imshow(imgs[j, i].cpu().numpy().transpose(1, 2, 0))
                axes[i, j].axis('off')

    plt.suptitle(tag)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, f'iteration_{iteration}.png'), bbox_inches='tight', dpi=120)
    else:
        plt.show()