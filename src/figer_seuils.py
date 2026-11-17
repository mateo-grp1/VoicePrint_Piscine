# src/figer_seuils.py
# Dernier réglage du projet : les SEUILS de la démo, pour tenir "au plus 1 % d'imposteurs acceptés"
# sur TOUTES les validations disponibles.
#
# Règle : pour chaque durée d'enrôlement (3, 5, 10 s), on prend le seuil le plus PRUDENT (le plus haut) parmi :
#   - validation cv26 (banc v3, 28 personnes) : test 3 s et test jusqu'à 10 s
#   - validation cv27 (290 personnes) : test 3 s et jusqu'à 10 s, deux définitions des voix proches
# Tous mesurés sur des tests propres, voix proches, avec la moyenne ACTUELLE de la démo.
#
# Ne change QUE les seuils : moyenne soustraite et calibration restent celles du banc v3 (vérifiées sur cv27).
# L'ancien fichier est gardé : models/reference_demo_v3.npz
#
# Prérequis : results/banc_v3.csv (eval_v3.py) et results/validation_cv27.csv (eval_cv27val.py)
# À lancer depuis la racine : python src/figer_seuils.py
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path("models/reference_demo.npz")
v3 = pd.read_csv("results/banc_v3.csv")
v27 = pd.read_csv("results/validation_cv27.csv")

with np.load(REF) as r:
    ref = {k: r[k].copy() for k in r.files}
anciens = {int(k): float(v) for k, v in json.loads(str(ref["seuils"])).items()}

lignes, seuils = [], {}
for d in [3, 5, 10]:
    candidats = {}
    for test in ["3s", "max10s"]:
        r = v3[(v3["enrôlement"] == f"{d} s") & (v3["test"] == test) & (v3["condition"] == "propre")]
        candidats[f"cv26, test {test}"] = float(r["seuil FA1%"].iloc[0])
        r = v27[(v27["enrôlement"] == f"{d} s") & (v27["test"] == test) & (v27["condition"] == "propre")]
        candidats[f"cv27 quart, test {test}"] = float(r["seuil FA1% cv27 quart"].iloc[0])
        candidats[f"cv27 ≤14Hz, test {test}"] = float(r["seuil FA1% cv27 ≤14Hz"].iloc[0])
    seuils[d] = max(candidats.values())
    source = max(candidats, key=candidats.get)
    lignes.append({"enrôlement": f"{d} s", **{k: round(v, 3) for k, v in candidats.items()},
                   "seuil retenu": round(seuils[d], 3), "fixé par": source, "ancien": round(anciens[d], 3)})

tab = pd.DataFrame(lignes)
pd.set_option("display.width", 250)
print("=== Seuils à 1 % d'imposteurs (voix proches, test propre) : on garde le plus prudent ===")
print(tab.T.to_string(header=False))
tab.to_csv("results/seuils_finaux.csv", index=False)

# Sauvegarde de l'ancien fichier, puis mise à jour des seuils UNIQUEMENT
if not Path("models/reference_demo_v3.npz").exists():
    shutil.copy(REF, "models/reference_demo_v3.npz")
ref["seuils"] = np.array(json.dumps({str(k): v for k, v in seuils.items()}))
ref["seuil"] = np.array(seuils[3])
ref["banc"] = np.array("v3 + cv27")
np.savez(REF, **ref)

a, b = float(ref["calib_a"]), float(ref["calib_b"])
print("\nSeuils de la démo : " + " | ".join(f"{d} s : {anciens[d]:.3f} -> {seuils[d]:.3f}" for d in seuils))
print(f"Confiance affichée au seuil 3 s : {1 / (1 + np.exp(-(a * seuils[3] + b))):.0%}")
print(f"Démo mise à jour : {REF} (ancienne version : models/reference_demo_v3.npz)")
print("Moyenne soustraite et calibration : inchangées. Configuration FIGÉE -> test final.")