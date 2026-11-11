# src/baseline_cosine.py
# Baseline : similarité cosinus entre vecteurs MFCC, sans puis avec normalisation.
# Évaluée sur les paires de VALIDATION. À lancer depuis la racine : python src/baseline_cosine.py
import numpy as np
import pandas as pd
from evaluate import evaluate, enregistrer

d = np.load("data/features/mfcc40.npz")
X, paths = d["X"], d["paths"]
idx = {p: i for i, p in enumerate(paths)}               # chemin -> numéro de ligne dans X

meta = pd.read_csv("data/metadata.csv")
train_idx = [idx[p] for p in meta.loc[meta["split"] == "train", "wav_path"]]
paires = pd.read_csv("data/pairs/pairs_val.csv")

def scores_cosinus(X):
    A = X[[idx[p] for p in paires["file1"]]]
    B = X[[idx[p] for p in paires["file2"]]]
    return (A * B).sum(axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))

# Version 1 : vecteurs bruts
res_brut = evaluate(scores_cosinus(X), paires["label"], nom="mfcc_cos_brut")
enregistrer("MFCC40 + cosinus (brut)", "val", res_brut)

# Version 2 : normalisation avec moyenne / écart-type du TRAIN uniquement
mu = X[train_idx].mean(axis=0)
sd = X[train_idx].std(axis=0) + 1e-8          # + petit nombre pour ne jamais diviser par 0
Xn = (X - mu) / sd
res_norm = evaluate(scores_cosinus(Xn), paires["label"], nom="mfcc_cos_norm")
enregistrer("MFCC40 + cosinus (normalisé train)", "val", res_norm, notes="ablation : normalisation")

print(pd.DataFrame([res_brut, res_norm], index=["brut", "normalisé"]).round(3))