import torch, torch.nn as nn, torch.nn.functional as F

def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))

def deconv(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

class ConvVAE(nn.Module):
    """VAE convolutionnel pour images 128x128 : 5 convolutions stride 2 (128 -> 4)."""
    def __init__(self, z_dim=128):
        super().__init__()
        self.enc = nn.Sequential(conv(3, 32), conv(32, 64), conv(64, 128),
                                 conv(128, 256), conv(256, 256))
        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_lv = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_dec = nn.Linear(z_dim, 256 * 4 * 4)
        self.dec = nn.Sequential(deconv(256, 256), deconv(256, 128), deconv(128, 64),
                                 deconv(64, 32), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 256, 4, 4))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.decode(z), mu, lv

def vae_loss(x, xr, mu, lv, beta=1e-2):
    """Erreur de reconstruction (somme par image) + beta * KL."""
    n = x.shape[0]
    rec = F.mse_loss(xr, x, reduction="sum") / n
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / n
    return rec + beta * kl, rec, kl