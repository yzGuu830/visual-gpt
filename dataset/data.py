from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from .cub200 import CUB200
from .stanforddogs import StanfordDogs


def make_dl(data_name='mnist', 
            data_path='~/data',
            train_bsz=256, 
            val_bsz=256,
            img_resolution=[256,256],
            **dl_kwargs):
    
    if data_name == 'cifar10':
        transform = transforms.ToTensor()
        train_ds = datasets.CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform)
        valid_ds = datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform)

    elif data_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(tuple(img_resolution)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            ])
        train_ds = datasets.ImageNet(root='~/data/imagenet', split='train', download=True, transform=transform)
        valid_ds = datasets.ImageNet(root='~/data/imagenet', split='val', download=True, transform=transform)

    elif data_name == 'stfdogs':
        transform = transforms.Compose([
            transforms.Resize(tuple(img_resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            ])
        train_ds = StanfordDogs(root=data_path, split='train', transform=transform, download=True)
        valid_ds = StanfordDogs(root=data_path, split='val', transform=transform, download=True)
    
    elif data_name == 'cub200':
        transform = transforms.Compose([
            transforms.Resize(tuple(img_resolution)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            ])
        train_ds = CUB200(root=data_path, split='train', transform=transform)
        valid_ds = CUB200(root=data_path, split='val', transform=transform)

    else:
        raise ValueError('Invalid dataset name')

    
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=train_bsz, shuffle=True, **dl_kwargs),
        'val': DataLoader(valid_ds, batch_size=val_bsz, shuffle=False, **dl_kwargs)
    }

    print(
        '[{}] Data Loaded!\n    Trainset: #-of-batches {}, image shape {}\n    Validation: #-of-batches {}, image shape {}\n'.format(
            data_name, len(dls['train']), next(iter(dls['train']))[0].shape, len(dls['val']), next(iter(dls['val']))[0].shape)
        )
    return dataloaders


if __name__ == "__main__":

    dls = make_dl('stfdogs', '../data', datrain_bsz=16, val_bsz=16)

    X, y = next(iter(dls['val']))
    print(X.shape, y.shape)