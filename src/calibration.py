# src/calibration.py
# Jour 5, étape 2 : transformer le score brut en probabilité lisible ("97 % que ce soit la même personne").
#  - Une régression logistique apprend, sur les paires de VALIDATION (3 s / 3 s, moyenne soustraite),
#    la correspondance score -> probabilité. Classes équilibrées : la probabilité suppose "une chance sur deux"
#    au départ, ce qui est le cas le plus simple à expliquer.
#  - Vérification honnête : on apprend sur la moitié des locuteurs, on vérifie sur l'autre moitié.
#  - Sauvegarde tout ce dont la démo v1 a besoin dans models/reference_demo.npz :
#    moyenne à soustraire, seuil, paramètres de calibration, écarts habituels pour l'explication.
# À lancer depuis la racine : python src/calibration.py
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, preparer, construire_reference, seuil_validation

SEUIL = seuil_validation()

bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
paires = bench[bench["duree"] == "3s"].reset_index(drop=True)
train = clips_train()
emb = calculer_empreintes(sorted(set(paires[["file1", "file2"]].values.ravel()) | set(train["wav_path"])),
                          charger_modele())
mu, _ = construire_reference(emb, train, moyenne=False)
A = preparer(np.stack([emb[f] for f in paires["file1"]]), mu)
B = preparer(np.stack([emb[f] for f in paires["file2"]]), mu)
paires["score"] = (A * B).sum(axis=1)


def apprendre(d):
    lr = LogisticRegression(class_weight="balanced").fit(d[["score"]], d["label"])
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def proba(s, a, b):
    return 1 / (1 + np.exp(-(a * np.asarray(s) + b)))


# 1. Vérification : apprendre sur une moitié des locuteurs, vérifier sur l'autre
locuteurs = np.array(sorted(set(paires["spk1"])))
moitie = set(np.random.default_rng(42).permutation(locuteurs)[:len(locuteurs) // 2])
dans_a = paires["spk1"].isin(moitie) & paires["spk2"].isin(moitie)
dans_b = ~paires["spk1"].isin(moitie) & ~paires["spk2"].isin(moitie)
a, b = apprendre(paires[dans_a])
verif = paires[dans_b].copy()
verif["p"] = proba(verif["score"], a, b)
verif["tranche"] = pd.cut(verif["p"], [0, 0.1, 0.5, 0.9, 1.0], include_lowest=True)
print("=== Vérification sur des locuteurs non vus par la calibration ===")
print("(une probabilité annoncée de 90 à 100 % doit correspondre à une vraie personne presque à chaque fois)")
print(verif.groupby("tranche", observed=True).agg(paires=("label", "size"),
                                                 vraie_personne_en_realite=("label", "mean")).round(2).to_string())

# 2. Calibration finale : apprise sur toutes les paires de validation
a, b = apprendre(paires)
print(f"\nCalibration finale : probabilité = 1 / (1 + exp(-({a:.2f} × score + {b:.2f})))")
print("\n=== Score brut -> probabilité affichée ===")
for s in sorted([0.1, 0.2, 0.3, SEUIL, 0.4, 0.5, 0.6, 0.7]):
    marque = "  <- seuil de la démo" if s == SEUIL else ""
    print(f"  score {s:.3f} -> {proba(s, a, b):6.1%}{marque}")

# 3. Tout ce que la démo v1 doit connaître, dans un seul fichier (models/ est exclu de GitHub)
intra = pd.read_csv("results/ecart_intra_locuteur.csv", index_col=0)["ecart_intra"].to_dict()
np.savez("models/reference_demo.npz", moyenne=mu, seuil=SEUIL, calib_a=a, calib_b=b,
         ecart_intra=json.dumps(intra))
print("\nRéglages de la démo enregistrés dans models/reference_demo.npz")