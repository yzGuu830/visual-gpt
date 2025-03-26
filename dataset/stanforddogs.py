import os
import torch
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import random

class StanfordDogs(Dataset):
    def __init__(self, root, split='train', transform=None, download=True):
        self.root = root
        self.split = split
        self.transform = transform
        self.download = download
        self.dataset_dir = os.path.join(self.root, 'StanfordDogs')
        
        if self.download:
            self._download_data()
        
        images_dir = os.path.join(self.dataset_dir, 'Images')
        self.classes = sorted(os.listdir(images_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.data = self._load_data()
    
    def _download_data(self):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)
            print("Downloading Stanford Dogs dataset...")

            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = downloaded / total_size * 100 if total_size > 0 else 0
                print(f"\rDownloading: {percent:.2f}%", end="")
            
            images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
            images_tar_path = os.path.join(self.dataset_dir, 'images.tar')
            print("Downloading images...")
            urllib.request.urlretrieve(images_url, images_tar_path, reporthook=progress_hook)
            print("Extracting images...")
            with tarfile.open(images_tar_path, 'r:*') as tar:
                tar.extractall(path=self.dataset_dir)
            
            annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
            annotations_tar_path = os.path.join(self.dataset_dir, 'annotation.tar')
            print("Downloading annotations...")
            urllib.request.urlretrieve(annotations_url, annotations_tar_path, reporthook=progress_hook)
            print("Extracting annotations...")
            with tarfile.open(annotations_tar_path, 'r:*') as tar:
                tar.extractall(path=self.dataset_dir)

            print("Stanford Dogs dataset downloaded and extracted.")
        else:
            print("Stanford Dogs dataset already exists.")
            
    def _load_data(self):
        data = []
        images_dir = os.path.join(self.dataset_dir, 'Images')
        split_file = os.path.join(self.dataset_dir, f'{self.split}_list.txt')

        if not os.path.exists(split_file):
            self._create_train_val_split()
            print("Created train_list.txt and val_list.txt split files for the first time")

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                # Expected format: "Images/<breed>/<filename>.jpg"
                img_path = os.path.join(self.dataset_dir, line)
                breed = os.path.basename(os.path.dirname(img_path))
                label = self.class_to_idx[breed]
                data.append((img_path, label))
        else:
            raise FileNotFoundError(f"Split file {split_file} not found")
        
        return data

    def _create_train_val_split(self):
        images_dir = os.path.join(self.dataset_dir, 'Images')
        train_list = []
        test_list = []
        random.seed(42)
        train_ratio = 0.95
        for breed in sorted(os.listdir(images_dir)):
            breed_dir = os.path.join(images_dir, breed)
            if not os.path.isdir(breed_dir):
                continue
            images = [os.path.join("Images", breed, fname) 
                    for fname in os.listdir(breed_dir) 
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_list.extend(images[:split_idx])
            test_list.extend(images[split_idx:])

        with open(os.path.join(self.dataset_dir, 'train_list.txt'), 'w') as f:
            f.write('\n'.join(train_list))
        with open(os.path.join(self.dataset_dir, 'val_list.txt'), 'w') as f:
            f.write('\n'.join(test_list))

    def _parse_annotation(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find('object')
            if obj is not None:
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    return (xmin, ymin, xmax, ymax)
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
        return None

    def _get_annotation_path(self, img_path):
        annotation_path = img_path.replace(
            os.path.join(self.dataset_dir, 'Images'),
            os.path.join(self.dataset_dir, 'Annotation')
        )
        annotation_path = annotation_path.rsplit('.', 1)[0]
        return annotation_path

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert('RGB')
        annotation_path = self._get_annotation_path(img_path)
        if os.path.exists(annotation_path):
            bbox = self._parse_annotation(annotation_path)
            if bbox is not None:
                image = image.crop(bbox)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor([label], dtype=torch.long)


if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = StanfordDogsDataset(root='data', split='train', transform=transform, download=True)
    print(f"Dataset size: {len(dataset)}")
    sample_image, sample_label = dataset[0]
    print(f"Sample image size: {sample_image.size()}, Label: {sample_label}")