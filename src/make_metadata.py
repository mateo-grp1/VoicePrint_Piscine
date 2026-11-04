# src/make_metadata.py
# Crée data/metadata.csv : une ligne par enregistrement Common Voice gardé.
# À lancer depuis la racine du projet : python src/make_metadata.py
import pandas as pd
from pathlib import Path

CV_DIR = Path("data/raw/cv26_fr/fr-FR")   # où sont clips/ et les .tsv
SORTIE = Path("data/metadata.csv")
MIN_CLIPS = 5                              # on garde les locuteurs avec au moins 5 clips

# 1. Lire les trois fichiers officiels et les mettre bout à bout
cols = ["client_id", "path", "age", "gender"]
morceaux = []
for split in ["train", "dev", "test"]:
    df = pd.read_csv(CV_DIR / f"{split}.tsv", sep="\t", usecols=cols, quoting=3)
    df["split_officiel"] = split           # on garde l'info, même si on refera notre propre découpage
    morceaux.append(df)
tout = pd.concat(morceaux, ignore_index=True)

# 2. Garder seulement les locuteurs avec assez de clips
n_clips = tout.groupby("client_id")["path"].transform("size")
tout = tout[n_clips >= MIN_CLIPS].reset_index(drop=True)

# 3. Remplacer le long identifiant anonyme par un identifiant court (cv_000, cv_001...)
ids = sorted(tout["client_id"].unique())   # le tri donne toujours le même ordre
correspondance = {cid: f"cv_{i:03d}" for i, cid in enumerate(ids)}
tout["speaker_id"] = tout["client_id"].map(correspondance)

# 4. Construire le tableau final
meta = pd.DataFrame({
    "file_path": [(CV_DIR / "clips" / p).as_posix() for p in tout["path"]],
    "speaker_id": tout["speaker_id"],
    "corpus": "commonvoice_26_fr-FR",
    "language": "fr",
    "gender": tout["gender"],
    "age": tout["age"],
    "split_officiel": tout["split_officiel"],
})

# 5. Vérifications
manquants = [f for f in meta["file_path"] if not Path(f).exists()]
print("Fichiers introuvables :", len(manquants))
print("Locuteurs :", meta["speaker_id"].nunique(), "| Clips :", len(meta))

meta.to_csv(SORTIE, index=False)
print("Écrit :", SORTIE)