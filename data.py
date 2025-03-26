import os
import json
import glob
import torch
from pathlib import Path
from PIL import Image

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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
        
        # valid_ds.filename = valid_ds.filename[:320]
        # valid_ds.attr = valid_ds.attr[:320]
        # print(train_ds[0][0].shape, train_ds[0][0].min(), train_ds[0][0].max())

    elif data_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1])), 
            transforms.ToTensor(), 
            ])
        train_ds = ImageNet100(root='../data/imagenet100', split='train', transform=transform)
        valid_ds = ImageNet100(root='../data/imagenet100', split='val', transform=transform)

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


from torch.utils.data import Dataset
class ImageNet100(Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        
        labels_dict = json.load(open(os.path.join(Path(root).expanduser(), 'labels100.json'), "r"))
        self.label_dict = self.preprocess_labels(labels_dict)
        
        self.transform = transform
        self.root_dir = os.path.join(Path(root).expanduser(), split)
        self.image_paths = glob.glob(os.path.join(self.root_dir, '*/*.JPEG'))

    def __len__(self):
        return len(self.image_paths)
    
    def preprocess_labels(self, labels_dict):
        label_dict = {}
        for k, v in labels_dict.items():
            label_dict[k] = v[0] # only use the first label

        self.classes = set([label for label in label_dict.values()])
        classes2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for img_class_id, label in label_dict.items():
            label_dict[img_class_id] = classes2idx.get(label)
        return label_dict
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        img_class_id = image_path.split("/")[-2]
        label = torch.tensor([self.label_dict[img_class_id]], dtype=torch.long)
        return image, label


class Coco2017Custom(Dataset):
    classes = ['car', 'bird', 'cat', 'dog']
    def __init__(self, root, split, transform=None):
        self.root_dir = os.path.join(Path(root).expanduser(), split)
        if not os.path.exists(self.root_dir):
            self.download(Path(root).expanduser().parent)

        self.transform = transform
        self.image_paths = glob.glob(os.path.join(self.root_dir, 'data/*.jpg'))

        labels_dict = json.load(open(os.path.join(self.root_dir, 'labels.json'), "r"))
        self.label_dict = self.preprocess_labels(labels_dict)

    def __len__(self):
        return len(self.image_paths)
    
    def download(self, path):
        print("Downloading COCO 2017 Customized dataset...")
        from huggingface_hub import hf_hub_download
        import tarfile
        
        repo_id = "Tracygu/CoCo"
        filename = "coco2017custom.tar.gz"
        tar_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=path)

        print(f"Extracting {filename}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=path, )

    def preprocess_labels(self, labels_dict):
        """preprocess mutli-labels to single-label"""
        classes2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        label_dict = {}
        for img_id, labels in labels_dict.items():
            label_dict[img_id] = classes2idx.get([label for label in self.classes if label in labels][0])
        return label_dict

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        img_id = image_path.split("/")[-1].split(".")[0]
        label = torch.tensor(self.label_dict[img_id], dtype=torch.long)
        return image, label

if __name__ == "__main__":

    dls = make_dl('coco2017custom', train_bsz=16, val_bsz=16)

    X, y = next(iter(dls['val']))
    print(X.shape, y.shape)