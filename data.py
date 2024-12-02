from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def make_dl(data_name='mnist', 
            train_bsz=256, 
            val_bsz=256,
            **dl_kwargs):
    
    if data_name == 'mnist': # res. 1 X 28 X 28 ; values 0 ~ 1
        transform = transforms.Compose([
            transforms.ToTensor(), 
            ])
        train_ds = datasets.MNIST(root='~/data/mnist', train=True, download=True, transform=transform)
        valid_ds = datasets.MNIST(root='~/data/mnist', train=False, download=True, transform=transform) 
    
    elif data_name == 'cifar10': # res. 3 X 32 X 32 ; values 0 ~ 1 
        transform = transforms.Compose([
            transforms.ToTensor(), 
            ])
        train_ds = datasets.CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform)
        valid_ds = datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform)

    elif data_name == 'celeba': # res. 3 X 218 X 178 ; values 0 ~ 1
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor(), 
            ])
        train_ds = datasets.CelebA(root='~/data/celeba', split='train', download=True, transform=transform)
        valid_ds = datasets.CelebA(root='~/data/celeba', split='valid', download=True, transform=transform)
        
        # valid_ds.filename = valid_ds.filename[:320]
        # valid_ds.attr = valid_ds.attr[:320]
        # print(train_ds[0][0].shape, train_ds[0][0].min(), train_ds[0][0].max())

    elif data_name == 'imagenet': # res. 3 X 224 X 224 ; values 0 ~ 1
        transform = transforms.Compose([
            transforms.Resize((64, 64)), 
            transforms.ToTensor(), 
            ])
        train_ds = datasets.ImageNet(root='~/data/imagenet', split='train', download=True, transform=transform)
        valid_ds = datasets.ImageNet(root='~/data/imagenet', split='val', download=True, transform=transform)
        

    dls = {'train': DataLoader(train_ds, batch_size=train_bsz, shuffle=True, **dl_kwargs),
           'val': DataLoader(valid_ds, batch_size=val_bsz, shuffle=False, **dl_kwargs)}

    print('Data Loaded!\nTrainset: #batches={}, input shape {}\nValidation: #batches={}, input shape {}\n'.format(
            len(dls['train']), next(iter(dls['train']))[0].shape, len(dls['val']), next(iter(dls['val']))[0].shape )
        )
    return dls


def make_inf_dl(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if __name__ == "__main__":

    make_dl('celeba')