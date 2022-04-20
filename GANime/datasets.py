import os
from PIL import Image
from torch.utils.data import Dataset

class ImageOnlyDataset(Dataset):
    def __init__(self, src_dir, transform):
        self.src_dir = src_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(src_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = os.path.join(self.src_dir, self.imgs[idx])
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        tensor_image = self.transform(img)
        return tensor_image