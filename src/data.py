import os, glob, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def list_train_val(root, cat, val_ratio=0.2, seed=42):
    """Split des images normales d'entraînement : train / val_normal."""
    files = sorted(glob.glob(f"{root}/{cat}/train/good/*.png"))
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_ratio)
    to_items = lambda fs: [(f, 0, None) for f in fs]
    return to_items(files[n_val:]), to_items(files[:n_val])

def list_test(root, cat):
    """Test officiel : (chemin, label 0/1, chemin du masque ou None)."""
    items = []
    for d in sorted(os.listdir(f"{root}/{cat}/test")):
        for f in sorted(glob.glob(f"{root}/{cat}/test/{d}/*.png")):
            if d == "good":
                items.append((f, 0, None))
            else:
                m = f"{root}/{cat}/ground_truth/{d}/{os.path.basename(f)[:-4]}_mask.png"
                items.append((f, 1, m))
    return items

class MVTecDataset(Dataset):
    def __init__(self, items, img_size=128, imagenet_norm=False):
        self.items, self.size = items, img_size
        tf = [T.Resize((img_size, img_size)), T.ToTensor()]
        if imagenet_norm:
            tf.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.tf = T.Compose(tf)
        self.mask_tf = T.Compose([T.Resize((img_size, img_size),
                                  interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, mpath = self.items[i]
        img = self.tf(Image.open(path).convert("RGB"))   # screw est en niveaux de gris !
        if mpath is None:
            mask = torch.zeros(1, self.size, self.size)
        else:
            mask = (self.mask_tf(Image.open(mpath).convert("L")) > 0).float()
        return img, label, mask