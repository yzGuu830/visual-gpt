from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from dataset.coco2017custom import Coco2017Custom
from dataset.cub200 import CUB200
from dataset.stanforddogs import StanfordDogs

def make_dl(data_name='mnist', 
            train_bsz=256, 
            val_bsz=256,
            img_size=(64, 64),
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

    # elif data_name == 'imagenet':
    #     transform = transforms.Compose([
    #         transforms.Resize((img_size[0], img_size[1])), 
    #         transforms.ToTensor(), 
    #         ])
    #     train_ds = ImageNetCustom(root='../data/imagenet', split='train', transform=transform)
    #     valid_ds = ImageNetCustom(root='../data/imagenet', split='val', transform=transform)
    elif data_name == 'stanforddogs':
        transform = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1])), 
            transforms.ToTensor(), 
            ])
        train_ds = StanfordDogs(root='../data', split='train', transform=transform)
        valid_ds = StanfordDogs(root='../data', split='val', transform=transform)
    
    elif data_name == 'cub200':
        transform = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1])), 
            transforms.ToTensor(), 
            ])
        train_ds = CUB200(root='../data/cub200/CUB_200_2011', split='train', transform=transform)
        valid_ds = CUB200(root='../data/cub200/CUB_200_2011', split='val', transform=transform)

    elif data_name == 'coco2017custom':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            ])
        train_ds = Coco2017Custom(root='~/data/coco2017custom', split='train', transform=transform)
        valid_ds = Coco2017Custom(root='~/data/coco2017custom', split='val', transform=transform)

    dls = {'train': DataLoader(train_ds, batch_size=train_bsz, shuffle=True, **dl_kwargs),
           'val': DataLoader(valid_ds, batch_size=val_bsz, shuffle=False, **dl_kwargs)}

    print('[{}] Data Loaded!\nTrainset: #batches={}, input shape {}\nValidation: #batches={}, input shape {}\n'.format(
            data_name, len(dls['train']), next(iter(dls['train']))[0].shape, len(dls['val']), next(iter(dls['val']))[0].shape )
        )
    return dls


def make_inf_dl(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if __name__ == "__main__":

    dls = make_dl('stanforddogs', train_bsz=16, val_bsz=16)

    X, y = next(iter(dls['val']))
    print(X.shape, y.shape)