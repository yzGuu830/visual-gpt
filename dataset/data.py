from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from .cub200 import CUB200
from .stanforddogs import StanfordDogs


def make_dataloaders(data_name='mnist', 
                     data_path='~/data',
                     train_bsz=256, 
                     val_bsz=256,
                     img_resolution=[256,256],
                     **dl_kwargs):
    
    if data_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5, 0.5)
            ])
        train_ds = datasets.CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform)
        valid_ds = datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform)

    elif data_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        valid_ds = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    elif data_name == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(img_resolution),
            transforms.ToTensor(),
        ])
        train_ds = datasets.CelebA(data_path,
            split='train', download=True, transform=transform)
        valid_ds = datasets.CelebA(data_path,
            split='valid', download=True, transform=transform)

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
        '[{}] data loaded!\n    trainset: #-of-batches {}, image shape {}\n    validation: #-of-batches {}, image shape {}\n'.format(
            data_name, len(dataloaders['train']), next(iter(dataloaders['train']))[0].shape, len(dataloaders['val']), next(iter(dataloaders['val']))[0].shape)
        )
    return dataloaders


if __name__ == "__main__":

    dls = make_dataloaders('cifar10', '../data', train_bsz=16, val_bsz=16)

    X, y = next(iter(dls['val']))
    print(X.shape, y.shape)
    print(X.min(), X.max(), X.mean(), X.std())