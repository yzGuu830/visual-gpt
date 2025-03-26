import os
import json
import glob
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


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