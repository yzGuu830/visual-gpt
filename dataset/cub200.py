import torch
from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path

class CUB200(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.root_dir = Path(root).expanduser()
        self.transform = transform
        self.split = split

        self.id2path = {}
        with open(self.root_dir / "images.txt", "r") as f:
            for line in f:
                image_id, image_name = line.strip().split()
                self.id2path[int(image_id)] = image_name

        self.path2id = {v: k for k, v in self.id2path.items()}

        self.bboxes = {}
        with open(self.root_dir / "bounding_boxes.txt", "r") as f:
            for line in f:
                image_id, x, y, w, h = map(float, line.strip().split())
                self.bboxes[int(image_id)] = (x, y, w, h)

        self.image_paths = []
        with open(self.root_dir / "custom_train_test_split.txt", "r") as f:
            for line in f:
                image_id, is_train = map(int, line.strip().split())
                if (split == 'train' and is_train) or (split == 'val' and not is_train):
                    self.image_paths.append(self.id2path[image_id])

        self.class2idx = {}
        with open(self.root_dir / "classes.txt", "r") as f:
            for line in f:
                idx, cls_name = line.strip().split()
                self.class2idx[cls_name] = int(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_rel_path = self.image_paths[idx]
        image_path = self.root_dir / "images" / image_rel_path
        image = Image.open(image_path).convert("RGB")

        image_id = self.path2id[image_rel_path]

        x, y, w, h = self.bboxes[image_id]
        left, upper, right, lower = int(x), int(y), int(x + w), int(y + h)
        image = image.crop((left, upper, right, lower))

        if self.transform:
            image = self.transform(image)

        class_name = image_rel_path.split("/")[0]
        label = torch.tensor([self.class2idx[class_name]], dtype=torch.long)
        return image, label