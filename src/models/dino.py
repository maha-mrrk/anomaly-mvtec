import torch
import torch.nn.functional as F

class DinoDetector:
    """ViT DINO gelé + banque mémoire de patchs normaux + score k-NN par patch."""
    def __init__(self, arch="dino_vits8", img_size=224, device="cuda"):
        self.device, self.img_size = device, img_size
        self.model = torch.hub.load("facebookresearch/dino:main", arch,
                                    trust_repo=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.patch = 8 if arch.endswith("8") else 16
        self.grid = img_size // self.patch          # 28 pour vits8, 14 pour vits16
        self.bank = None

    def patch_features(self, x):
        """Tokens de patchs du dernier bloc, sans le CLS : (B, N, D)."""
        return self.model.get_intermediate_layers(x, n=1)[0][:, 1:]

    @torch.no_grad()
    def fit(self, loader):
        """Banque mémoire = tous les patchs des images d'entraînement."""
        feats = [self.patch_features(x.to(self.device)).flatten(0, 1) for x, _, _ in loader]
        self.bank = torch.cat(feats)                # (N_patchs_total, D)

    def knn_dist(self, q, k=1, chunk=1024):
        out = []
        for i in range(0, len(q), chunk):           # par blocs : évite de saturer la mémoire GPU
            d = torch.cdist(q[i:i + chunk], self.bank)
            out.append(d.topk(k, dim=1, largest=False).values.mean(1))
        return torch.cat(out)

    def anomaly_map(self, x, k=1):
        """(B,3,H,W) -> (B,H,W). Reste différentiable (utile pour l'étape adversariale)."""
        f = self.patch_features(x)
        B = f.shape[0]
        d = self.knn_dist(f.flatten(0, 1), k).reshape(B, 1, self.grid, self.grid)
        return F.interpolate(d, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)[:, 0]