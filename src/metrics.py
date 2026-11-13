import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from skimage.measure import label as cc_label
from src.scoring import smooth_maps, image_scores_from_maps

def image_auroc(labels, scores):
    return roc_auc_score(labels, scores)

def pixel_auroc(masks, maps):
    return roc_auc_score(masks.reshape(-1) > 0.5, maps.reshape(-1))

def aupro(masks, maps, fpr_max=0.3, n_thr=100):
    """Aire sous la courbe PRO (overlap par région) jusqu'à FPR=fpr_max. Optionnel."""
    masks = masks > 0.5
    regions = []
    for i, m in enumerate(masks):
        if m.any():
            lab, n = cc_label(m, return_num=True)
            regions += [(i, lab == r) for r in range(1, n + 1)]
    neg = ~masks
    n_neg = neg.sum()
    fprs, pros = [], []
    for t in np.linspace(maps.min(), maps.max(), n_thr):
        pred = maps >= t
        fprs.append((pred & neg).sum() / n_neg)
        pros.append(np.mean([(pred[i] & r).sum() / r.sum() for i, r in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    o = np.argsort(fprs); fprs, pros = fprs[o], pros[o]
    keep = fprs <= fpr_max
    f = np.append(fprs[keep], fpr_max)
    p = np.append(pros[keep], np.interp(fpr_max, fprs, pros))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return trapz(p, f) / fpr_max

def evaluate(labels, masks, maps, sigma=4, score_mode="max", with_aupro=False):
    maps = smooth_maps(maps, sigma) if sigma else maps
    scores = image_scores_from_maps(maps, score_mode)
    out = {"img_auroc": image_auroc(labels, scores),
           "pix_auroc": pixel_auroc(masks, maps)}
    if with_aupro:
        out["aupro"] = aupro(masks, maps)
    return out

def log_result(csv_path, **row):
    """Ajoute une ligne au tableau de résultats (crée le fichier si besoin)."""
    pd.DataFrame([row]).to_csv(csv_path, mode="a",
                               header=not os.path.exists(csv_path), index=False)