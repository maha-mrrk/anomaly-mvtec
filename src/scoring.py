import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

def smooth_maps(maps, sigma=4):
    """Lissage gaussien de chaque carte (maps : N,H,W)."""
    return np.stack([gaussian_filter(m, sigma=sigma) for m in maps])

def image_scores_from_maps(maps, mode="max", topk=0.01):
    """Score image : max de la carte, ou moyenne du top-k% des pixels."""
    flat = maps.reshape(len(maps), -1)
    if mode == "max":
        return flat.max(axis=1)
    k = max(1, int(flat.shape[1] * topk))
    return np.sort(flat, axis=1)[:, -k:].mean(axis=1)

@torch.no_grad()
def predict_maps(map_fn, dataset, batch_size=16, device="cuda"):
    """Applique map_fn à tout le dataset. Renvoie maps, labels, masks (numpy)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    maps, labels, masks = [], [], []
    for x, y, m in loader:
        a = map_fn(x.to(device))                       # (B,H,W)
        maps.append(a.detach().cpu().numpy())
        labels.append(y.numpy())
        masks.append(m[:, 0].numpy())
    return np.concatenate(maps), np.concatenate(labels), np.concatenate(masks)