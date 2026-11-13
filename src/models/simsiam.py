"""SimSiam (T8) : modele, augmentations, dataset a 2 vues, perte.
A placer dans : anomaly-mvtec/src/models/simsiam.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]


def simsiam_augment(size):
    """Augmentations douces : PAS de flip / rotation (l'orientation compte, ex. cable)
    et rien qui ressemble a un defaut (pas de cutout, pas de gros bruit)."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])


class TwoViews(Dataset):
    """Renvoie 2 vues augmentees de la meme image normale.
    Les images sont pre-chargees en RAM (redimensionnees un peu au-dessus de `size`)
    pour ne pas re-decoder des PNG 1024x1024 a chaque epoch."""

    def __init__(self, items, size):
        pre = int(size * 1.15)
        self.imgs = [Image.open(it[0]).convert("RGB").resize((pre, pre), Image.BILINEAR)
                     for it in items]
        self.tf = simsiam_augment(size)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        return self.tf(img), self.tf(img)


def _projector(in_dim, dim):
    return nn.Sequential(
        nn.Linear(in_dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
        nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
        nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim),
    )


class SimSiam(nn.Module):
    """ResNet-18 + projecteur MLP 3 couches + predicteur MLP 2 couches."""

    def __init__(self, pretrained=False, dim=512, pred_hidden=128):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # on coupe le backbone en deux pour pouvoir recuperer layer3 (cartes par patch)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.layer3 = r.layer3
        self.layer4 = r.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = _projector(512, dim)
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_hidden, bias=False), nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True), nn.Linear(pred_hidden, dim),
        )

    def features(self, x):
        """Renvoie (carte layer3 [B,256,H/16,W/16], embedding global [B,512])."""
        f3 = self.layer3(self.stem(x))
        emb = self.pool(self.layer4(f3)).flatten(1)
        return f3, emb

    def forward(self, x1, x2):
        z1 = self.projector(self.features(x1)[1])
        z2 = self.projector(self.features(x2)[1])
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def D(p, z):
    # stop-gradient sur z : c'est CA qui evite l'effondrement
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


def simsiam_loss(p1, p2, z1, z2):
    return 0.5 * D(p1, z2) + 0.5 * D(p2, z1)
