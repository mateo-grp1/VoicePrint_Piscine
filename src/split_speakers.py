# src/split_speakers.py
# Sépare les LOCUTEURS (pas les enregistrements) en train / val / test (60/20/20),
# vérifie automatiquement qu'il n'y a aucune fuite, puis ajoute une colonne "split" à metadata.csv.
# À lancer depuis la racine du projet : python src/split_speakers.py
import numpy as np
import pandas as pd
from pathlib import Path

META = Path("data/metadata.csv")
SEED = 42          # graine fixe : même découpage à chaque relance
P_TRAIN, P_VAL = 0.6, 0.2   # le reste (0.2) va en test

meta = pd.read_csv(META)

# 1. Liste des locuteurs uniques, triée puis mélangée avec la graine
locuteurs = np.array(sorted(meta["speaker_id"].unique()))
rng = np.random.default_rng(SEED)
rng.shuffle(locuteurs)

# 2. Découpage de la liste des locuteurs
n = len(locuteurs)
n_train = round(n * P_TRAIN)
n_val = round(n * P_VAL)
groupes = {
    "train": locuteurs[:n_train],
    "val":   locuteurs[n_train:n_train + n_val],
    "test":  locuteurs[n_train + n_val:],
}

# 3. Chaque enregistrement reçoit le groupe de SON locuteur
groupe_de = {loc: g for g, liste in groupes.items() for loc in liste}
meta["split"] = meta["speaker_id"].map(groupe_de)

# 4. Test automatique : aucune fuite possible
tr, va, te = set(groupes["train"]), set(groupes["val"]), set(groupes["test"])
assert not (tr & va) and not (tr & te) and not (va & te), "FUITE : un locuteur est dans deux groupes !"
assert meta["split"].notna().all(), "Un enregistrement n'a pas de groupe !"
assert (meta.groupby("speaker_id")["split"].nunique() == 1).all(), "Un locuteur a des clips dans deux groupes !"
print("Test de fuite : OK (aucun locuteur partagé)")

# 5. Résumé par groupe
resume = meta.groupby("split").agg(
    locuteurs=("speaker_id", "nunique"),
    clips=("wav_path", "size"),
    heures=("duration_s", lambda d: round(d.sum() / 3600, 2)),
)
resume["max_clips_1_locuteur"] = meta.groupby(["split", "speaker_id"]).size().groupby("split").max()
print(resume)

meta.to_csv(META, index=False)
print("Mis à jour :", META)