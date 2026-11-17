# src/pires_paires.py
# Cas d'échec : liste les PIRES essais du test final, pour les écouter.
#   - les 5 vraies personnes refusées avec les scores les plus bas ;
#   - les 5 imposteurs avec les scores les plus hauts (les plus proches d'être acceptés).
# Configuration : enrôlement 3 s, test jusqu'à 10 s, propre (le cas principal de la démo).
# Ne recalcule rien : relit le cache du test final (data/embeddings/test_final.npz) et la configuration figée.
# Prérequis : python src/test_final.py déjà lancé.
# À lancer depuis la racine : python src/pires_paires.py
import json

import numpy as np
import pandas as pd

from normalisation import preparer

GRAINE, N_TESTS_MAX, N_ENROL_MIN, N = 42, 10, 2, 5
D_ENROL, MODE, COND = 3, "max10s", "propre"
rng = np.random.default_rng(GRAINE)

# Même répartition enrôlement / tests que test_final.py (même graine, même ordre)
meta = pd.read_csv("data/metadata.csv")
tst = meta[meta["split"] == "test"]
locs = sorted(tst["speaker_id"].unique())
plan_enrol, plan_tests = {}, []
for s in locs:
    fichiers = list(rng.permutation(sorted(tst.loc[tst["speaker_id"] == s, "wav_path"])))
    n_tests = min(N_TESTS_MAX, len(fichiers) - N_ENROL_MIN)
    plan_tests += [(f, s) for f in fichiers[:n_tests]]
    plan_enrol[s] = fichiers[n_tests:]

with np.load("models/reference_demo.npz") as r:
    mu = r["moyenne"].copy()
    seuil = {int(k): float(v) for k, v in json.loads(str(r["seuils"])).items()}[D_ENROL]
with np.load("data/embeddings/test_final.npz", allow_pickle=False) as d:
    emb = dict(zip(d["cles"], d["emb"]))

E = preparer(np.stack([emb[f"enrol|{D_ENROL}|{s}"] for s in locs]), mu)
T = preparer(np.stack([emb[f"test|{MODE}|{COND}|{f}"] for f, _ in plan_tests]), mu)
S = E @ T.T
spk_test = np.array([s for _, s in plan_tests])

lignes = []
for i, s in enumerate(locs):
    for j, (f, st) in enumerate(plan_tests):
        lignes.append({"personne enrôlée": s, "personne testée": st, "vraie personne": s == st,
                       "score": float(S[i, j]), "fichier de test": f,
                       "fichiers d'enrôlement": " | ".join(plan_enrol[s])})
t = pd.DataFrame(lignes)
refus = t[t["vraie personne"]].nsmallest(N, "score").assign(type="vraie personne refusée")
imp = t[~t["vraie personne"]].nlargest(N, "score").assign(type="imposteur au score élevé")
res = pd.concat([refus, imp])
res["accepté (seuil démo)"] = res["score"] >= seuil
res["cause probable (à remplir après écoute)"] = ""
res.to_csv("results/pires_paires.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)
print(f"Seuil de la démo (enrôlement {D_ENROL} s) : {seuil:.3f}\n")
for nom, g in res.groupby("type", sort=False):
    print(f"=== {nom} ===")
    print(g[["personne enrôlée", "personne testée", "score", "accepté (seuil démo)", "fichier de test"]]
          .round(3).to_string(index=False))
    print()
print("Détail (avec les fichiers d'enrôlement à écouter) : results/pires_paires.csv")
print("Écoute chaque fichier de test ET l'enrôlement correspondant, puis note la cause probable dans le CSV.")