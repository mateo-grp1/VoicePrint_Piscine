# src/eval_mini_corpus.py
# Jour 4 : test "vraie vie" avec la configuration retenue
#   enrôlement = 3 premières secondes de parole (un seul bloc), test = 3 s de parole,
#   empreintes ECAPA avec soustraction de la moyenne du train, seuil choisi sur la VALIDATION.
# Chaque personne du mini-corpus est enrôlée à son tour, puis comparée à :
#   - ses propres prises de test (elle doit être acceptée),
#   - les prises des autres personnes du mini-corpus (même pièce, même micro : imposteurs difficiles),
#   - les extraits de 3 s des 28 locuteurs Common Voice de validation (imposteurs en nombre).
# Aucun réglage n'est fait ici : on vérifie seulement.
# À lancer depuis la racine : python src/eval_mini_corpus.py
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, preparer, construire_reference, seuil_validation

SR, DUREE = 16000, 3


def parole(f):
    """Charge un fichier et retire les silences du début et de la fin."""
    y, _ = librosa.load(f, sr=SR)
    return librosa.effects.trim(y, top_db=30)[0]


def empreinte(model, y):
    with torch.no_grad():
        return model.encode_batch(torch.tensor(y).unsqueeze(0)).squeeze().numpy()


SEUIL = seuil_validation()
model = charger_modele()

# Moyenne du train (à soustraire) et imposteurs Common Voice (extraits 3 s de validation, déjà en cache)
train = clips_train()
bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
cv = sorted(set(bench.loc[bench["duree"] == "3s", ["file1", "file2"]].values.ravel()))
emb = calculer_empreintes(sorted(set(cv) | set(train["wav_path"])), model)
mu, _ = construire_reference(emb, train, moyenne=False)
E_cv = preparer(np.stack([emb[f] for f in cv]), mu)

# Mini-corpus : enrôlements (3 premières secondes de parole) et tests (3 s prises au milieu de la parole)
meta = pd.read_csv("data/mini_corpus/metadata.csv")
personnes = sorted(meta["personne"].unique())
enrol, tests = {}, []
for _, r in meta.iterrows():
    y = parole(r["fichier"])
    if r["condition"] == "enrolement":
        enrol[r["personne"]] = preparer(empreinte(model, y[:DUREE * SR]), mu)[0]
    else:
        debut = max(0, (len(y) - DUREE * SR) // 2)
        tests.append({"personne": r["personne"], "condition": r["condition"], "langue": r["langue"],
                      "emb": preparer(empreinte(model, y[debut:debut + DUREE * SR]), mu)[0]})
print(f"Mini-corpus : {len(personnes)} personne(s), {len(tests)} prises de test | "
      f"imposteurs Common Voice : {len(cv)} extraits | seuil (validation) = {SEUIL:.3f}")

# Scores
lignes = []
for p, ref in enrol.items():
    for t in tests:
        s = float(ref @ t["emb"])
        lignes.append({"enrôlé": p, "parle": t["personne"], "condition": t["condition"], "langue": t["langue"],
                       "vrai": t["personne"] == p, "score": s, "accepté": s >= SEUIL})
res = pd.DataFrame(lignes)
Path("results").mkdir(exist_ok=True)
res.to_csv("results/mini_corpus_scores.csv", index=False)

vrais = res[res["vrai"]]
print("\n=== La vraie personne parle (doit être ACCEPTÉE) ===")
print(vrais[["enrôlé", "condition", "langue", "score", "accepté"]].round(3).to_string(index=False))

print("\n=== Imposteurs (doivent être REFUSÉS) ===")
bilan = []
for p, ref in enrol.items():
    proches = res[(res["enrôlé"] == p) & ~res["vrai"]]
    s_cv = E_cv @ ref
    bilan.append({"enrôlé": p,
                  "proches : prises": len(proches),
                  "proches : score max": proches["score"].max() if len(proches) else np.nan,
                  "proches : acceptés": int(proches["accepté"].sum()),
                  "CV : extraits": len(s_cv), "CV : score max": s_cv.max(),
                  "CV : acceptés": int((s_cv >= SEUIL).sum())})
print(pd.DataFrame(bilan).round(3).to_string(index=False))

n_cv = sum(b["CV : extraits"] for b in bilan)
print(f"\nBilan : vraie personne refusée {int((~vrais['accepté']).sum())}/{len(vrais)} fois | "
      f"proches acceptés {sum(b['proches : acceptés'] for b in bilan)}/{sum(b['proches : prises'] for b in bilan)} | "
      f"Common Voice acceptés {sum(b['CV : acceptés'] for b in bilan)}/{n_cv}")