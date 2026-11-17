# src/eval_cv27val.py
# ÉTUDE COMPLÉMENTAIRE (ne modifie PAS la démo) : grande validation Common Voice 27.0.
# 290 personnes jamais vues par le système final, préparées exactement comme l'application (notebook 05),
# comparées aux 28 personnes du banc v3 (cv26).
#
# Ce script ne touche ni à models/reference_demo.npz, ni à results/resultats.csv.
# Il écrit seulement results/validation_cv27.csv et results/validation_cv27_config_actuelle.csv.
#
#   0. Contrôles : même recette (preparation.py) et même ECAPA que sur ce PC
#   1. La configuration ACTUELLE de la démo (moyenne cv26 préparée démo + seuils 0,416 / 0,447 / 0,482)
#      mesurée telle quelle sur les 290 nouvelles personnes
#   2. Les moyennes possibles (dont un mélange cv26 + cv27), résultats rendus SÉPARÉMENT pour cv26 et cv27
#   3. Deux définitions des "voix proches" cv27 : le quart le plus proche (règle des bancs v2/v3)
#      et un écart de hauteur de 14 Hz au plus (même difficulté que cv26)
#   4. Seuils à 1 % d'imposteurs mesurés séparément sur cv26 et sur cv27
#   5. Calibration vérifiée sur des essais ÉQUILIBRÉS (autant de vraies personnes que d'imposteurs)
#
# Prérequis : data/cv27/banc_cv27val.npz, banc_cv27val_info.csv, split_cv27.csv, emb_cv27_train.npz,
#             controle_ecapa.npy ; data/embeddings/banc_v3.npz et ecapa.npz ; data/pairs/pairs_val_v3.csv
# À lancer depuis la racine : python src/eval_cv27val.py
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import evaluate
from normalisation import clips_train, construire_reference, preparer, unitaire

pd.set_option("display.width", 220)
DUREES_ENROL = [3, 5, 10]
TESTS = ["3s", "max10s"]
CONDITIONS = ["propre", "ambiance", "brouhaha"]
ECART_F0_MAX = 14.0          # Hz : écart maximal des "voix proches" du banc cv26
CV27 = Path("data/cv27")
rng = np.random.default_rng(42)

# ---------- Chargements ----------
with np.load(CV27 / "banc_cv27val.npz", allow_pickle=False) as d:
    emb27 = dict(zip(d["cles"], d["emb"]))
    controle_colab, recette_colab = d["controle"], str(d["recette"])
info = pd.read_csv(CV27 / "banc_cv27val_info.csv", dtype=str)
split = pd.read_csv(CV27 / "split_cv27.csv", dtype={"locuteur": str})
with np.load("data/embeddings/banc_v3.npz", allow_pickle=False) as d:
    emb26 = dict(zip(d["cles"], d["emb"]))
paires26 = pd.read_csv("data/pairs/pairs_val_v3.csv", keep_default_na=False)
with np.load("models/reference_demo.npz") as r:                     # lecture seule
    SEUILS_ACTUELS = {int(k): float(v) for k, v in json.loads(str(r["seuils"])).items()}
    MU_ACTUELLE = r["moyenne"].copy()

# ---------- 0. Contrôles ----------
print("=== 0) Contrôles ===")
recette_pc = hashlib.md5(Path("src/preparation.py").read_bytes()).hexdigest()
print(f"Recette preparation.py identique Colab / PC : {'oui ✅' if recette_pc == recette_colab else 'NON ⚠️'}")
controle_pc = np.load(CV27 / "controle_ecapa.npy")
cos = float(controle_pc @ controle_colab / (np.linalg.norm(controle_pc) * np.linalg.norm(controle_colab)))
print(f"Même ECAPA Colab / PC : cosinus {cos:.5f} {'✅' if cos > 0.999 else '⚠️'}")
val_locs = sorted(split.loc[split["groupe"] == "val", "locuteur"])
assert set(info["locuteur"]) <= set(val_locs), "Un locuteur hors validation dans le banc !"

# ---------- Moyennes ----------
train26 = clips_train()
with np.load("data/embeddings/ecapa.npz", allow_pickle=True) as d:
    ecapa_brut = dict(zip(d["paths"], d["emb"]))
mu_brut, _ = construire_reference(ecapa_brut, train26, moyenne=False)
E26_demo = unitaire(np.stack([emb26[f"train|{f}"] for f in train26["wav_path"]]))
mu_demo = E26_demo.mean(axis=0)
with np.load(CV27 / "emb_cv27_train.npz", allow_pickle=True) as d:
    train27 = set(split.loc[split["groupe"] == "train", "locuteur"].str[1:].astype(int))
    E27_train = unitaire(d["emb"][(d["version"] < 2) & np.isin(d["locuteur"], list(train27))])
mu_cv27 = E27_train.mean(axis=0)
mu_melange = (mu_demo + mu_cv27) / 2          # chaque corpus pèse autant, quel que soit son nombre de clips
assert np.allclose(mu_demo, MU_ACTUELLE, atol=1e-5), "La moyenne actuelle de la démo n'est pas 'cv26 préparée démo'"
MOYENNES = {"cv26 préparée démo (ACTUELLE)": mu_demo, "cv26 brute": mu_brut,
            "cv27 train": mu_cv27, "mélange cv26 + cv27": mu_melange}

# ---------- Essais cv27 : chaque enrôlement contre tous les clips de test ----------
tests27 = info[info["role"] == "test"].reset_index(drop=True)
spk_test, spk_enrol = tests27["locuteur"].to_numpy(), np.array(val_locs)
LABEL27 = spk_enrol[:, None] == spk_test[None, :]
f0 = split.set_index("locuteur")["f0_locuteur"]
ecart = np.abs(f0[spk_enrol].to_numpy()[:, None] - f0[spk_test].to_numpy()[None, :])
couples = sorted(combinations(val_locs, 2), key=lambda c: abs(f0[c[0]] - f0[c[1]]))
quart = {frozenset(c) for c in couples[:len(couples) // 4]}
PROCHES27 = {
    "quart le plus proche": np.array([[frozenset((a, b)) in quart for b in spk_test] for a in spk_enrol]),
    f"écart F0 ≤ {ECART_F0_MAX:.0f} Hz": ecart <= ECART_F0_MAX,
}
DIFF26 = paires26["type"].isin(["positive", "negative_difficile"]).to_numpy()
LABEL26 = paires26["label"].to_numpy()


def scores27(mu, d_enrol, mode, cond):
    E = preparer(np.stack([emb27[f"enrol|{d_enrol}|{l}"] for l in val_locs]), mu)
    T = preparer(np.stack([emb27[f"test|{mode}|{cond}|{c}"] for c in tests27["clip"]]), mu)
    return E @ T.T


def scores26(mu, d_enrol, mode, cond):
    A = preparer(np.stack([emb26[f"enrol|{d_enrol}|{f}|{x}"] for f, x in zip(paires26["file1"], paires26["exclu"])]), mu)
    B = preparer(np.stack([emb26[f"test|{mode}|{cond}|{f}"] for f in paires26["file2"]]), mu)
    return (A * B).sum(axis=1)


def taux(s, label, imposteurs, seuil):
    return (s[imposteurs] >= seuil).mean(), (s[label] < seuil).mean()


for nom, m in PROCHES27.items():
    print(f"Voix proches cv27 ({nom}) : {int(LABEL27.sum())} essais vraie personne, "
          f"{int((m & ~LABEL27).sum())} imposteurs proches")
print(f"Voix proches cv26 : {int(LABEL26.sum())} essais vraie personne, {int((DIFF26 & (LABEL26 == 0)).sum())} imposteurs proches")

# ---------- 1. La configuration actuelle, telle quelle ----------
print("\n=== 1) Configuration ACTUELLE de la démo (inchangée) mesurée sur les 290 nouvelles personnes, test propre ===")
lignes = []
for d_enrol in DUREES_ENROL:
    seuil = SEUILS_ACTUELS[d_enrol]
    for mode in TESTS:
        s27, s26 = scores27(mu_demo, d_enrol, mode, "propre"), scores26(mu_demo, d_enrol, mode, "propre")
        ligne = {"enrôlement": f"{d_enrol} s", "test": mode, "seuil actuel": seuil}
        for nom, m in PROCHES27.items():
            fa, frr = taux(s27, LABEL27, m & ~LABEL27, seuil)
            ligne[f"imposteurs acceptés cv27 ({nom})"] = fa
        ligne["vraies personnes refusées cv27"] = (s27[LABEL27] < seuil).mean()
        fa26, frr26 = taux(s26, LABEL26 == 1, DIFF26 & (LABEL26 == 0), seuil)
        ligne["imposteurs acceptés cv26"], ligne["vraies personnes refusées cv26"] = fa26, frr26
        lignes.append(ligne)
actuelle = pd.DataFrame(lignes)
Path("results").mkdir(exist_ok=True)
actuelle.to_csv("results/validation_cv27_config_actuelle.csv", index=False)
print(actuelle.round(3).to_string(index=False))

# ---------- 2 et 3. Moyennes et définitions des voix proches, cv26 et cv27 séparés ----------
print("\n=== 2) Moyennes possibles (enrôlement 3 s, moyenne sur 6 tests), cv26 et cv27 SÉPARÉS ===")
lignes = []
for nom_mu, mu in MOYENNES.items():
    for mode in TESTS:
        for cond in CONDITIONS:
            s27, s26 = scores27(mu, 3, mode, cond), scores26(mu, 3, mode, cond)
            r26 = evaluate(s26[DIFF26], LABEL26[DIFF26])
            ligne = {"moyenne": nom_mu, "minDCF cv26": r26["min_dcf"], "FRR@FA1% cv26": r26["frr_far1"]}
            for nom_p, m in PROCHES27.items():
                sel = m | LABEL27
                r27 = evaluate(s27[sel], LABEL27[sel])
                ligne[f"minDCF cv27 ({nom_p})"], ligne[f"FRR@FA1% cv27 ({nom_p})"] = r27["min_dcf"], r27["frr_far1"]
            lignes.append(ligne)
print(pd.DataFrame(lignes).groupby("moyenne", sort=False).mean().round(3).to_string())

# ---------- Détail, configuration actuelle de moyenne ----------
lignes = []
for d_enrol in DUREES_ENROL:
    for mode in TESTS:
        for cond in CONDITIONS:
            s27, s26 = scores27(mu_demo, d_enrol, mode, cond), scores26(mu_demo, d_enrol, mode, cond)
            r26 = evaluate(s26[DIFF26], LABEL26[DIFF26])
            ligne = {"enrôlement": f"{d_enrol} s", "test": mode, "condition": cond,
                     "EER cv26": r26["eer"], "FRR@FA1% cv26": r26["frr_far1"], "seuil FA1% cv26": r26["seuil_far1"]}
            for nom_p, m in PROCHES27.items():
                sel = m | LABEL27
                r27 = evaluate(s27[sel], LABEL27[sel])
                court = "quart" if nom_p.startswith("quart") else "≤14Hz"
                ligne[f"EER cv27 {court}"] = r27["eer"]
                ligne[f"FRR@FA1% cv27 {court}"] = r27["frr_far1"]
                ligne[f"seuil FA1% cv27 {court}"] = r27["seuil_far1"]
            lignes.append(ligne)
detail = pd.DataFrame(lignes)
detail.to_csv("results/validation_cv27.csv", index=False)
print("\n=== 3) Détail avec la moyenne ACTUELLE (seuils FA1% mesurés séparément sur chaque validation) ===")
print(detail.round(3).to_string(index=False))

# ---------- 4. Calibration actuelle, vérifiée sur des essais ÉQUILIBRÉS ----------
with np.load("models/reference_demo.npz") as r:
    a, b = float(r["calib_a"]), float(r["calib_b"])
s = scores27(mu_demo, 3, "max10s", "propre")
pos = s[LABEL27]
neg_tous = s[~LABEL27]
neg = rng.choice(neg_tous, size=len(pos), replace=False)            # autant d'imposteurs que de vraies personnes
cal = pd.DataFrame({"score": np.r_[pos, neg], "label": np.r_[np.ones(len(pos)), np.zeros(len(neg))]})
cal["p"] = 1 / (1 + np.exp(-(a * cal["score"] + b)))
cal["tranche"] = pd.cut(cal["p"], [0, 0.1, 0.5, 0.9, 1.0], include_lowest=True)
print(f"\n=== 4) Calibration ACTUELLE (a = {a:.2f}, b = {b:.2f}) vérifiée sur cv27, essais équilibrés ===")
print(cal.groupby("tranche", observed=True).agg(essais=("label", "size"),
                                               vraie_personne_en_realite=("label", "mean")).round(3).to_string())
print("\nRésultats écrits : results/validation_cv27.csv, results/validation_cv27_config_actuelle.csv")
print("La démo n'est pas modifiée (models/reference_demo.npz est seulement lu).")