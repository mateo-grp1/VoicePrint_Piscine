# src/eval_v3.py
# Évaluation sur le banc v3 (voix préparées exactement comme dans la démo), puis mise à jour de la démo.
#
#   1. Quelle moyenne soustraire ? cv26 brute (actuelle) / cv26 préparée comme la démo / cv27 (945 locuteurs)
#   2. La couche metric learning v2 re-testée à armes égales (sans réentraînement)
#   3. Tableau par durée d'enrôlement (3/5/10 s), durée de test (3 s / jusqu'à 10 s) et bruit
#   4. Seuils (1 % d'imposteurs acceptés, voix proches) et calibration, choisis sur la VALIDATION
#   5. Écrit models/reference_demo.npz (l'ancien est gardé en models/reference_demo_v1.npz)
# Prérequis : python src/build_bench_v3.py
# À lancer depuis la racine : python src/eval_v3.py
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from build_bench_v3 import BANC, CACHE, CONDITIONS, DUREES_ENROL, TESTS, cle_enrol, cle_test
from evaluate import enregistrer, evaluate
from normalisation import clips_train, construire_reference, preparer, unitaire

pd.set_option("display.width", 200)
paires = pd.read_csv(BANC, keep_default_na=False)
labels = paires["label"].to_numpy()
difficile = paires["type"].isin(["positive", "negative_difficile"]).to_numpy()
with np.load(CACHE, allow_pickle=False) as d:
    emb = dict(zip(d["cles"], d["emb"]))


def bruts(d_enrol, mode, cond):
    A = np.stack([emb[cle_enrol(d_enrol, f, x)] for f, x in zip(paires["file1"], paires["exclu"])])
    B = np.stack([emb[cle_test(mode, cond, f)] for f in paires["file2"]])
    return A, B


def scores(mu, d_enrol, mode, cond, couche=None):
    A, B = bruts(d_enrol, mode, cond)
    A, B = preparer(A, mu), preparer(B, mu)
    if couche is not None:
        with torch.no_grad():
            A = couche(torch.tensor(A, dtype=torch.float32)).numpy()
            B = couche(torch.tensor(B, dtype=torch.float32)).numpy()
    return (A * B).sum(axis=1)


def mesure(s, sel=difficile):
    return evaluate(s[sel], labels[sel])


# ---------- 1. Les trois moyennes possibles ----------
train = clips_train()
with np.load("data/embeddings/ecapa.npz", allow_pickle=True) as d:
    ecapa_brut = dict(zip(d["paths"], d["emb"]))
mu_brut, _ = construire_reference(ecapa_brut, train, moyenne=False)
mu_demo, _ = construire_reference({f: emb[f"train|{f}"] for f in train["wav_path"]}, train, moyenne=False)
with np.load("data/cv27/emb_cv27_train.npz", allow_pickle=True) as d:
    propres = d["emb"][d["version"] < 2]                       # versions propres uniquement
mu_cv27 = unitaire(propres).mean(axis=0)
MOYENNES = {"cv26 brute (actuelle)": mu_brut, "cv26 préparée démo": mu_demo, "cv27 (945 locuteurs)": mu_cv27}

print("=== 1) Quelle moyenne soustraire ? (enrôlement 3 s, voix proches, minDCF, plus bas = mieux) ===")
lignes = []
for nom, mu in MOYENNES.items():
    for mode in TESTS:
        for cond in CONDITIONS:
            r = mesure(scores(mu, 3, mode, cond))
            lignes.append({"moyenne": nom, "test": f"{mode} {cond}", "minDCF": r["min_dcf"],
                           "FRR@FA1%": r["frr_far1"], "EER": r["eer"]})
t = pd.DataFrame(lignes)
print(t.pivot_table(index="test", columns="moyenne", values="minDCF")[list(MOYENNES)].round(3).to_string())
print("\nFRR@FA1% :")
print(t.pivot_table(index="test", columns="moyenne", values="FRR@FA1%")[list(MOYENNES)].round(3).to_string())
moy = t.groupby("moyenne")[["minDCF", "FRR@FA1%"]].mean()
print("\nMoyenne sur les 6 tests :\n" + moy.round(3).to_string())
choix = moy["minDCF"].idxmin()
MU = MOYENNES[choix]
print(f"\n-> Moyenne retenue (plus petit minDCF moyen) : {choix}")


# ---------- 2. Couche metric learning v2, à armes égales ----------
class Diagonale(nn.Module):
    def __init__(self, d=192):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.normalize(x * self.w, dim=-1)


class ResiduelleRang(nn.Module):
    def __init__(self, d=192, r=32):
        super().__init__()
        self.bas = nn.Linear(d, r, bias=False)
        self.haut = nn.Linear(r, d)

    def forward(self, x):
        return F.normalize(x + self.haut(self.bas(x)), dim=-1)


print("\n=== 2) Couche metric learning v2 sur le banc v3 (moyenne cv26 brute, celle de son entraînement) ===")
lignes = []
for nom, classe in [("diagonale", Diagonale), ("residuelle_r32", ResiduelleRang)]:
    f = Path(f"models/couche_metric_v2_{nom}.pt")
    if not f.exists():
        continue
    couche = classe()
    couche.load_state_dict(torch.load(f, weights_only=False)["poids"])
    couche.eval()
    for mode in TESTS:
        for cond in CONDITIONS:
            r0, r1 = mesure(scores(mu_brut, 3, mode, cond)), mesure(scores(mu_brut, 3, mode, cond, couche))
            lignes.append({"couche": nom, "test": f"{mode} {cond}", "minDCF ECAPA": r0["min_dcf"],
                           "minDCF + couche": r1["min_dcf"], "FRR@FA1% ECAPA": r0["frr_far1"],
                           "FRR@FA1% + couche": r1["frr_far1"]})
if lignes:
    print(pd.DataFrame(lignes).round(3).to_string(index=False))

# ---------- 3. Tableau complet avec la moyenne retenue ----------
print(f"\n=== 3) Banc v3, moyenne {choix}, voix proches ===")
lignes = []
for d_enrol in DUREES_ENROL:
    for mode in TESTS:
        for cond in CONDITIONS:
            s = scores(MU, d_enrol, mode, cond)
            r = mesure(s)
            enregistrer(f"ECAPA + moyenne {choix} | banc v3 | enrôlement {d_enrol} s | test {mode} | {cond} | difficile",
                        "val_v3", r, notes="préparation identique à la démo (src/preparation.py)")
            lignes.append({"enrôlement": f"{d_enrol} s", "test": mode, "condition": cond, "EER": r["eer"],
                           "minDCF": r["min_dcf"], "FRR@FA1%": r["frr_far1"], "seuil FA1%": r["seuil_far1"]})
tab = pd.DataFrame(lignes)
Path("results").mkdir(exist_ok=True)
tab.to_csv("results/banc_v3.csv", index=False)
print(tab.round(3).to_string(index=False))

# ---------- 4. Seuils et calibration (validation uniquement) ----------
print("\n=== 4) Seuils de la démo (1 % d'imposteurs acceptés, voix proches, test propre) ===")
print("On garde le plus PRUDENT des deux tests (3 s ou jusqu'à 10 s) : au plus 1 % d'imposteurs, "
      "que la personne parle peu ou beaucoup.")
seuils = {}
for d_enrol in DUREES_ENROL:
    s3 = tab[(tab["enrôlement"] == f"{d_enrol} s") & (tab["test"] == "3s") & (tab["condition"] == "propre")]["seuil FA1%"].iloc[0]
    s10 = tab[(tab["enrôlement"] == f"{d_enrol} s") & (tab["test"] == "max10s") & (tab["condition"] == "propre")]["seuil FA1%"].iloc[0]
    seuils[d_enrol] = float(max(s3, s10))
    print(f"  enrôlement {d_enrol:2d} s : test 3 s {s3:.3f} | test jusqu'à 10 s {s10:.3f} -> seuil {seuils[d_enrol]:.3f}")

# Calibration : enrôlement 3 s, test comme la démo (jusqu'à 10 s), propre, toutes les paires
cal = paires[["spk1", "spk2", "label"]].copy()
cal["score"] = scores(MU, 3, "max10s", "propre")


def apprendre(d):
    lr = LogisticRegression(class_weight="balanced").fit(d[["score"]], d["label"])
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def proba(s, a, b):
    return 1 / (1 + np.exp(-(a * np.asarray(s) + b)))


locs = np.array(sorted(set(cal["spk1"])))
moitie = set(np.random.default_rng(42).permutation(locs)[:len(locs) // 2])
dans_a = cal["spk1"].isin(moitie) & cal["spk2"].isin(moitie)
dans_b = ~cal["spk1"].isin(moitie) & ~cal["spk2"].isin(moitie)
a, b = apprendre(cal[dans_a])
verif = cal[dans_b].copy()
verif["tranche"] = pd.cut(proba(verif["score"], a, b), [0, 0.1, 0.5, 0.9, 1.0], include_lowest=True)
print("\n=== Calibration vérifiée sur des locuteurs non vus par elle ===")
print(verif.groupby("tranche", observed=True).agg(paires=("label", "size"),
                                                 vraie_personne_en_realite=("label", "mean")).round(2).to_string())
a, b = apprendre(cal)
print(f"Calibration finale : probabilité = 1 / (1 + exp(-({a:.2f} × score + {b:.2f}))) | "
      f"au seuil 3 s ({seuils[3]:.3f}) : {proba(seuils[3], a, b):.0%}")

# ---------- 5. Mise à jour de la démo ----------
ref = Path("models/reference_demo.npz")
if ref.exists() and not Path("models/reference_demo_v1.npz").exists():
    shutil.copy(ref, "models/reference_demo_v1.npz")
intra = pd.read_csv("results/ecart_intra_locuteur.csv", index_col=0)["ecart_intra"].to_dict()
np.savez(ref, moyenne=MU, seuil=seuils[3], seuils=json.dumps({str(k): v for k, v in seuils.items()}),
         calib_a=a, calib_b=b, ecart_intra=json.dumps(intra), banc="v3", moyenne_source=choix)
print(f"\nDémo mise à jour : {ref} (ancienne version gardée : models/reference_demo_v1.npz)")
print("Tableau complet : results/banc_v3.csv")