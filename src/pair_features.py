# src/pair_features.py
# Outils partagés : charger les MFCC normalisés et fabriquer les entrées des paires.
import numpy as np
import pandas as pd

def charger_mfcc_normalises():
    """Charge les vecteurs MFCC et les normalise avec la moyenne / l'écart-type du TRAIN uniquement."""
    d = np.load("data/features/mfcc40.npz")
    X, paths = d["X"], d["paths"]
    idx = {p: i for i, p in enumerate(paths)}
    meta = pd.read_csv("data/metadata.csv")
    train_idx = [idx[p] for p in meta.loc[meta["split"] == "train", "wav_path"]]
    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0) + 1e-8
    return (X - mu) / sd, idx

def entrees_paires(Xn, idx, paires, mode="diff_prod"):
    """Pour chaque paire : |a-b| (mode 'diff') ou |a-b| suivi de a*b (mode 'diff_prod')."""
    A = Xn[[idx[p] for p in paires["file1"]]]
    B = Xn[[idx[p] for p in paires["file2"]]]
    if mode == "diff":
        return np.abs(A - B)
    return np.hstack([np.abs(A - B), A * B])