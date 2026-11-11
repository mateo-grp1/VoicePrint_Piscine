# src/ecapa_bench.py
# Jour 3 : évalue ECAPA sur le banc d'essai v2, par durée et par type de paires négatives.
# Réutilise le cache d'empreintes : seuls les nouveaux extraits courts sont calculés.
# À lancer depuis la racine : python src/ecapa_bench.py
import pandas as pd
from evaluate import evaluate, enregistrer
from ecapa_eval import charger_modele, calculer_empreintes, scores_cosinus

bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
chemins = sorted(set(bench[["file1", "file2"]].values.ravel()))
emb = calculer_empreintes(chemins, charger_modele())
bench["score"] = scores_cosinus(bench, emb)

lignes = []
for duree in ["entier", "3s", "2s"]:
    b = bench[bench["duree"] == duree]
    for cas, negs in [("hasard", ["negative_hasard"]),
                      ("difficile", ["negative_difficile"]),
                      ("tout", ["negative_hasard", "negative_difficile"])]:
        sel = b[(b["type"] == "positive") | b["type"].isin(negs)]
        nom = f"ecapa_v2_{duree}_{cas}"
        r = evaluate(sel["score"], sel["label"], nom=nom if cas == "tout" else None)
        enregistrer(f"ECAPA + cosinus | {duree} | négatives {cas}", "val_v2", r)
        lignes.append({"durée": duree, "négatives": cas, "nb paires": len(sel),
                       "EER": r["eer"], "minDCF": r["min_dcf"], "seuil EER": r["seuil_eer"]})

print(pd.DataFrame(lignes).round(3).to_string(index=False))
print("\nScore moyen par type de paire et par durée :")
print(bench.groupby(["duree", "type"])["score"].mean().unstack().round(3))