# src/make_pairs.py
# Crée les paires positives / négatives (50/50) pour train, val et test,
# à l'intérieur de chaque groupe, puis les sauvegarde dans data/pairs/.
# À lancer depuis la racine du projet : python src/make_pairs.py
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

META = Path("data/metadata.csv")
OUT = Path("data/pairs")
SEED = 42
POS_PAR_LOC = {"train": 50, "val": 20, "test": 20}   # plafond de paires positives par locuteur

meta = pd.read_csv(META)
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

def paires_positives(df, k):
    """Pour chaque locuteur : jusqu'à k paires de 2 clips différents, tirées au hasard."""
    lignes = []
    for loc, g in df.groupby("speaker_id"):
        fichiers = sorted(g["wav_path"])
        combos = list(itertools.combinations(fichiers, 2))     # toutes les paires possibles
        idx = rng.choice(len(combos), size=min(k, len(combos)), replace=False)
        for i in sorted(idx):
            a, b = combos[i]
            lignes.append((a, b, loc, loc, 1))
    return lignes

def paires_negatives(df, n):
    """n paires de 2 locuteurs différents : on tire d'abord les personnes, puis un clip de chacune."""
    par_loc = {loc: sorted(g["wav_path"]) for loc, g in df.groupby("speaker_id")}
    locs = sorted(par_loc)
    vues, lignes = set(), []
    while len(lignes) < n:
        l1, l2 = rng.choice(locs, size=2, replace=False)
        a, b = str(rng.choice(par_loc[l1])), str(rng.choice(par_loc[l2]))
        cle = tuple(sorted((a, b)))
        if cle in vues:            # on évite les doublons
            continue
        vues.add(cle)
        lignes.append((a, b, str(l1), str(l2), 0))
    return lignes

for split in ["train", "val", "test"]:
    df = meta[meta["split"] == split]
    pos = paires_positives(df, POS_PAR_LOC[split])
    neg = paires_negatives(df, len(pos))                       # autant de négatives que de positives
    paires = pd.DataFrame(pos + neg, columns=["file1", "file2", "spk1", "spk2", "label"])
    paires = paires.sample(frac=1, random_state=SEED).reset_index(drop=True)   # on mélange l'ordre

    # Vérifications automatiques
    locs_split = set(df["speaker_id"])
    assert set(paires["spk1"]) | set(paires["spk2"]) <= locs_split, "FUITE : locuteur d'un autre groupe !"
    assert ((paires["spk1"] == paires["spk2"]) == (paires["label"] == 1)).all(), "Étiquette incohérente !"
    assert (paires["file1"] != paires["file2"]).all(), "Un clip comparé à lui-même !"

    max_pos = paires[paires["label"] == 1].groupby("spk1").size().max()
    print(f"{split:5s} | paires : {len(paires)} | positives : {paires['label'].sum()} "
          f"| négatives : {(paires['label'] == 0).sum()} | locuteurs : {len(locs_split)} "
          f"| max positives pour 1 locuteur : {max_pos}")
    paires.to_csv(OUT / f"pairs_{split}.csv", index=False)

print("Paires sauvegardées dans", OUT)