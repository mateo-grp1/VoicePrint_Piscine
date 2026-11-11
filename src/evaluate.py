# src/evaluate.py
# Fonction d'évaluation UNIQUE, utilisée par toutes les approches.
# Jour 3 : ajout du taux de faux rejets à 1 % de fausses acceptations (frr_far1),
# et enregistrer() accepte désormais de nouvelles colonnes sans décaler le tableau.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

RESULTS = Path("results")


def evaluate(scores, labels, nom=None, p_target=0.05, c_miss=1, c_fa=1, far_cible=0.01):
    """Renvoie EER, minDCF, AUC, faux rejets à 1 % de fausses acceptations, et les seuils correspondants.
    Trace la ROC si nom est donné."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    fpr, tpr, seuils = roc_curve(labels, scores)   # fpr = fausses acceptations
    fnr = 1 - tpr                                  # fnr = faux rejets

    # EER : le point où les deux erreurs sont égales
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[i] + fnr[i]) / 2

    # minDCF : coût pondéré des deux erreurs, normalisé, au meilleur seuil
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    dcf = dcf / min(c_miss * p_target, c_fa * (1 - p_target))
    j = np.argmin(dcf)

    # Faux rejets quand on n'accepte au plus que 1 % d'imposteurs (le réglage "démo")
    ok = np.where(fpr <= far_cible)[0]
    k = ok[np.argmax(tpr[ok])]

    res = {"eer": eer, "seuil_eer": seuils[i], "min_dcf": dcf[j], "seuil_min_dcf": seuils[j],
           "auc": roc_auc_score(labels, scores), "frr_far1": fnr[k], "seuil_far1": seuils[k]}

    if nom:
        RESULTS.mkdir(exist_ok=True)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"{nom} (EER = {eer:.1%})")
        plt.plot([0, 1], [0, 1], "--", color="grey", label="hasard")
        plt.xlabel("Taux de fausses acceptations")
        plt.ylabel("Taux de vraies acceptations")
        plt.title("Courbe ROC")
        plt.legend()
        plt.savefig(RESULTS / f"roc_{nom}.png", dpi=120, bbox_inches="tight")
        plt.close()
    return res


def enregistrer(nom, split, res, notes=""):
    """Ajoute une ligne au tableau comparatif results/resultats.csv (les nouvelles colonnes sont ajoutées proprement)."""
    RESULTS.mkdir(exist_ok=True)
    fichier = RESULTS / "resultats.csv"
    ligne = pd.DataFrame([{"approche": nom, "paires": split, **res, "notes": notes}])
    if fichier.exists():
        ligne = pd.concat([pd.read_csv(fichier), ligne], ignore_index=True)
    ligne.to_csv(fichier, index=False)