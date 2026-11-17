"""
Sépare (une fois pour toutes) les 945 locuteurs Common Voice 27.0 en deux groupes :
  - "train" (~655) : ne serviront qu'à entraîner (metric learning, moyenne cv27)
  - "val"   (290)  : grande validation, préparée comme l'application (notebook 05)

Règles :
  - séparation PAR LOCUTEUR, graine 42 ;
  - validation équilibrée : 145 voix graves + 145 voix aiguës (F0 < / >= 165 Hz) ;
  - les locuteurs cv27 qui sont AUSSI dans le train cv26 vont obligatoirement en "train" :
    leurs voix ont servi à calculer la moyenne soustraite, ce serait une petite fuite en validation.

Lancer depuis la racine :  python src/separer_cv27.py
Sortie : data/cv27/split_cv27.csv  ->  à déposer dans Mon Drive/voiceprint/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_cv27 import DOSSIER, table_cv26  # noqa: E402

GRAINE = 42
N_VAL_PAR_GROUPE = 145          # 145 graves + 145 aigus = 290 locuteurs de validation
SEUIL_F0 = 165.0

sel = pd.read_csv(DOSSIER / "selection_cv27.csv", dtype=str)
loc = sel.drop_duplicates("locuteur")[["locuteur", "client_id", "f0_locuteur"]].copy()
loc["f0_locuteur"] = loc["f0_locuteur"].astype(float)

# Locuteurs cv27 déjà présents dans cv26 (par client_id ou par noms de clips) : jamais en validation
cv26 = table_cv26()
ids_cv26 = set(cv26["client_id"].dropna())
loc_par_clips = set(sel.loc[sel["clip"].isin(set(cv26["clip"])), "client_id"])
dans_cv26 = loc["client_id"].isin(ids_cv26 | loc_par_clips)
print(f"Locuteurs cv27 : {len(loc)} | déjà présents dans cv26 (forcés en train) : {int(dans_cv26.sum())}")

rng = np.random.default_rng(GRAINE)
libres = loc[~dans_cv26]
graves = libres[libres["f0_locuteur"] < SEUIL_F0]["locuteur"].to_numpy()
aigus = libres[libres["f0_locuteur"] >= SEUIL_F0]["locuteur"].to_numpy()
val = set(rng.choice(np.sort(graves), N_VAL_PAR_GROUPE, replace=False)) | \
      set(rng.choice(np.sort(aigus), N_VAL_PAR_GROUPE, replace=False))

loc["groupe"] = np.where(loc["locuteur"].isin(val), "val", "train")
assert not (loc["groupe"].eq("val") & dans_cv26).any(), "FUITE : un locuteur cv26 est en validation !"
assert loc.groupby("locuteur")["groupe"].nunique().max() == 1

sortie = DOSSIER / "split_cv27.csv"
loc[["locuteur", "groupe", "f0_locuteur"]].sort_values("locuteur").to_csv(sortie, index=False)

print(loc.groupby("groupe").agg(locuteurs=("locuteur", "size"),
                                graves=("f0_locuteur", lambda f: int((f < SEUIL_F0).sum())),
                                aigus=("f0_locuteur", lambda f: int((f >= SEUIL_F0).sum())),
                                f0_mediane=("f0_locuteur", "median")).round(0).to_string())
print(f"\nÉcrit : {sortie}  ->  à déposer dans Mon Drive/voiceprint/")