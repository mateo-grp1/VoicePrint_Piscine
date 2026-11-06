# src/baseline_classifiers.py
# Classifieurs simples sur les paires : entraînés sur pairs_train, évalués sur pairs_val.
# À lancer depuis la racine : python src/baseline_classifiers.py
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from evaluate import evaluate, enregistrer
from pair_features import charger_mfcc_normalises, entrees_paires

Xn, idx = charger_mfcc_normalises()
p_train = pd.read_csv("data/pairs/pairs_train.csv")
p_val = pd.read_csv("data/pairs/pairs_val.csv")

# nom affiché, nom de fichier (sans caractères interdits sous Windows), entrées, modèle
modeles = [
    ("LogReg |a-b|",       "logreg_diff",      "diff",
     make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("LogReg |a-b| + a*b", "logreg_diff_prod", "diff_prod",
     make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("SVM RBF |a-b| + a*b", "svm_diff_prod",   "diff_prod",
     make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))),
]

lignes = []
for nom, fichier, mode, modele in modeles:
    X_tr = entrees_paires(Xn, idx, p_train, mode)
    X_va = entrees_paires(Xn, idx, p_val, mode)

    debut = time.time()
    modele.fit(X_tr, p_train["label"])                 # apprentissage sur le TRAIN
    duree = time.time() - debut

    r_tr = evaluate(modele.decision_function(X_tr), p_train["label"])
    r_va = evaluate(modele.decision_function(X_va), p_val["label"], nom=fichier)
    enregistrer(f"MFCC40 + {nom}", "val", r_va, notes=f"entraîné sur pairs_train en {duree:.1f} s")

    lignes.append({"approche": nom, "eer_train": r_tr["eer"], "eer_val": r_va["eer"],
                   "min_dcf_val": r_va["min_dcf"], "auc_val": r_va["auc"], "temps_s": duree})

print(pd.DataFrame(lignes).round(3).to_string(index=False))