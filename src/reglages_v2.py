# src/reglages_v2.py
# Jour 3, étape 2 : effet de la soustraction de moyenne et de l'AS-norm sur le banc d'essai v2.
# Un seul réglage change à la fois ; mêmes paires, même evaluate().
# À lancer depuis la racine : python src/reglages_v2.py
import numpy as np
import pandas as pd
from evaluate import evaluate, enregistrer
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, preparer, construire_reference, as_norm

bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
train = clips_train()
print(f"Référence : {train['speaker_id'].nunique()} locuteurs de TRAIN, {len(train)} clips")

# Empreintes : celles du banc sont déjà en cache, on ajoute celles du train
chemins = sorted(set(bench[["file1", "file2"]].values.ravel()) | set(train["wav_path"]))
emb = calculer_empreintes(chemins, charger_modele())

mu, cohorte_brute = construire_reference(emb, train, moyenne=False)
_, cohorte_centree = construire_reference(emb, train, moyenne=True)

configs = {
    "1. brut (cosinus)":            dict(moyenne=False, asnorm=False),
    "2. + soustraction moyenne":    dict(moyenne=True,  asnorm=False),
    "3. + AS-norm":                 dict(moyenne=False, asnorm=True),
    "4. + moyenne + AS-norm":       dict(moyenne=True,  asnorm=True),
}

lignes = []
for nom, c in configs.items():
    A = preparer(np.stack([emb[f] for f in bench["file1"]]), mu if c["moyenne"] else None)
    B = preparer(np.stack([emb[f] for f in bench["file2"]]), mu if c["moyenne"] else None)
    s = (A * B).sum(axis=1)
    if c["asnorm"]:
        s = as_norm(s, A, B, cohorte_centree if c["moyenne"] else cohorte_brute)
    bench["score"] = s

    for duree in ["entier", "3s", "2s"]:
        for cas, negs in [("difficile", ["negative_difficile"]),
                          ("tout", ["negative_hasard", "negative_difficile"])]:
            sel = bench[(bench["duree"] == duree) & ((bench["type"] == "positive") | bench["type"].isin(negs))]
            r = evaluate(sel["score"], sel["label"])
            enregistrer(f"ECAPA {nom} | {duree} | {cas}", "val_v2", r)
            lignes.append({"réglage": nom, "durée": duree, "négatives": cas,
                           "EER": r["eer"], "minDCF": r["min_dcf"], "FRR@FA1%": r["frr_far1"]})

t = pd.DataFrame(lignes)
for mesure in ["EER", "minDCF", "FRR@FA1%"]:
    print(f"\n=== {mesure} (plus bas = mieux) ===")
    print(t.pivot_table(index="réglage", columns=["durée", "négatives"], values=mesure)
          [["entier", "3s", "2s"]].round(3).to_string())