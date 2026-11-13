import os, random, subprocess
import numpy as np
import torch

REPO         = "/content/anomaly-mvtec"
DRIVE_ROOT   = "/content/drive/MyDrive/anomaly_mvtec"
ARCHIVES_DIR = f"{DRIVE_ROOT}/data"            # archives .tar.xz sur Drive
CKPT_DIR     = f"{DRIVE_ROOT}/checkpoints"     # poids et résultats (persistants)
DATA_ROOT    = "/content/mvtec"                # données extraites (disque local)
RESULTS_CSV  = f"{CKPT_DIR}/results.csv"
CATEGORIES   = ["carpet", "screw", "cable"]

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data(categories=CATEGORIES):
    """Extrait les archives depuis Drive vers le disque local (ignore ce qui existe déjà)."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for cat in categories:
        if not os.path.exists(f"{DATA_ROOT}/{cat}"):
            subprocess.run(["tar", "-xf", f"{ARCHIVES_DIR}/{cat}.tar.xz",
                            "-C", DATA_ROOT], check=True)
    print("Données prêtes :", [c for c in categories if os.path.exists(f"{DATA_ROOT}/{c}")])