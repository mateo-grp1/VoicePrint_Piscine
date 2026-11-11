# src/build_bench_v2.py
# Jour 3, étape 1 : un banc d'essai plus exigeant, construit UNIQUEMENT avec les locuteurs de validation.
#  - paires négatives "difficiles" : deux voix proches (même genre si connu + hauteur moyenne proche)
#  - extraits courts (2 s et 3 s) en plus des clips entiers
# Le fichier pairs_val.csv d'origine n'est PAS modifié (il reste la référence de la baseline).
# À lancer depuis la racine : python src/build_bench_v2.py
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from itertools import combinations
from pathlib import Path

GRAINE = 42
N_POS_PAR_LOC = 15          # paires positives max par locuteur
PART_PROCHES = 0.25         # le quart des couples de même genre dont la hauteur est la plus proche
DUREES_COURTES = [2, 3]     # secondes
SR = 16000
rng = np.random.default_rng(GRAINE)

# 1. Clips et locuteurs de validation, repris des paires existantes (aucun locuteur de train ou de test)
p_val = pd.read_csv("data/pairs/pairs_val.csv")
clips = pd.concat([p_val[["file1", "spk1"]].set_axis(["file", "spk"], axis=1),
                   p_val[["file2", "spk2"]].set_axis(["file", "spk"], axis=1)]).drop_duplicates("file")
par_loc = clips.groupby("spk")["file"].apply(sorted).to_dict()
print(f"{len(par_loc)} locuteurs de validation, {len(clips)} clips")

# 2. Genre (si la colonne existe et est remplie) et hauteur moyenne (F0) de chaque locuteur
genre = {}
meta = pd.read_csv("data/metadata.csv")
if "gender" in meta.columns:
    g = meta.set_index("wav_path")["gender"].dropna().to_dict()
    for spk, fichiers in par_loc.items():
        vals = [g[f] for f in fichiers if f in g]
        if vals:
            genre[spk] = pd.Series(vals).mode()[0]
print(f"Genre connu pour {len(genre)} locuteurs sur {len(par_loc)}")


def f0_mediane(fichiers, n_max=5):
    """Hauteur médiane de la voix (Hz) sur quelques clips, mesurée sur les seules parties voisées."""
    valeurs = []
    for f in fichiers[:n_max]:
        y, _ = librosa.load(f, sr=SR)
        f0, voise, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR)
        valeurs.extend(f0[voise])
    return float(np.median(valeurs)) if valeurs else np.nan


print("Calcul de la hauteur de chaque voix (quelques minutes)...")
f0 = {spk: f0_mediane(fichiers) for spk, fichiers in par_loc.items()}
Path("data/features").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"spk": list(f0), "f0_hz": list(f0.values()),
              "genre": [genre.get(s, "") for s in f0]}).to_csv("data/features/f0_locuteurs_val.csv", index=False)

# 3. Paires positives : deux clips différents d'un même locuteur
pos = []
for spk, fichiers in par_loc.items():
    combis = list(combinations(fichiers, 2))
    for i in rng.permutation(len(combis))[:N_POS_PAR_LOC]:
        pos.append((*combis[i], spk, spk, 1, "positive"))
n = len(pos)


def tirer_negatives(couples_loc, n, nom):
    """n paires négatives tirées dans une liste de couples de locuteurs."""
    out = []
    for k in rng.integers(0, len(couples_loc), n):
        a, b = couples_loc[k]
        out.append((rng.choice(par_loc[a]), rng.choice(par_loc[b]), a, b, 0, nom))
    return out


# 4. Négatives "au hasard" (comme avant) et négatives "difficiles" (voix proches)
tous = list(combinations(par_loc, 2))
meme_genre = [(a, b) for a, b in tous
              if a not in genre or b not in genre or genre[a] == genre[b]]
# On classe les couples par écart de hauteur et on garde le quart le plus proche
meme_genre.sort(key=lambda c: abs(f0[c[0]] - f0[c[1]]))
proches = meme_genre[:max(1, int(len(meme_genre) * PART_PROCHES))]
ecarts = [abs(f0[a] - f0[b]) for a, b in proches]
print(f"Couples de locuteurs : {len(tous)} au total, {len(proches)} jugés proches "
      f"(écart de hauteur entre {min(ecarts):.0f} et {max(ecarts):.0f} Hz)")

paires = pd.DataFrame(pos + tirer_negatives(tous, n, "negative_hasard")
                      + tirer_negatives(proches, n, "negative_difficile"),
                      columns=["file1", "file2", "spk1", "spk2", "label", "type"])
paires["duree"] = "entier"

# 5. Extraits courts : un morceau pris au milieu de chaque clip (même position pour toutes les méthodes)
versions = [paires]
for d in DUREES_COURTES:
    dossier = Path(f"data/processed/extraits_{d}s")
    dossier.mkdir(parents=True, exist_ok=True)
    nouveau = {}
    for f in pd.unique(paires[["file1", "file2"]].values.ravel()):
        y, _ = librosa.load(f, sr=SR)
        milieu = max(0, (len(y) - d * SR) // 2)
        cible = (dossier / Path(f).name).as_posix()
        sf.write(cible, y[milieu:milieu + d * SR], SR)
        nouveau[f] = cible
    v = paires.copy()
    v["file1"], v["file2"], v["duree"] = v["file1"].map(nouveau), v["file2"].map(nouveau), f"{d}s"
    versions.append(v)

bench = pd.concat(versions, ignore_index=True)
bench.to_csv("data/pairs/pairs_val_v2.csv", index=False)
print("\nBanc d'essai v2 enregistré : data/pairs/pairs_val_v2.csv")
print(bench.groupby(["duree", "type"]).size().unstack())