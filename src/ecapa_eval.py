# src/ecapa_eval.py
# Jour 2 : premier vrai EER d'ECAPA, sur EXACTEMENT les mêmes paires que la baseline.
# Étape 1 : calcule une fois l'empreinte de chaque clip utilisé dans les paires (mise en cache).
# Étape 2 : score chaque paire de validation par cosinus, puis evaluate() + enregistrer().
# À lancer depuis la racine du projet : python src/ecapa_eval.py
import time
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from evaluate import evaluate, enregistrer

CACHE = Path("data/embeddings/ecapa.npz")      # data/ est exclu de GitHub
FICHIERS_PAIRES = ["data/pairs/pairs_val.csv", "data/pairs/pairs_test.csv"]


def charger_modele():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/ecapa")


def calculer_empreintes(chemins, model):
    """Empreinte ECAPA (192 nombres) de chaque fichier. Les empreintes déjà calculées sont réutilisées."""
    deja = {}
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        deja = dict(zip(d["paths"], d["emb"]))

    a_faire = [c for c in chemins if c not in deja]
    print(f"{len(deja)} empreintes déjà en cache, {len(a_faire)} à calculer")
    debut = time.time()
    for n, chemin in enumerate(a_faire, 1):
        y, _ = librosa.load(chemin, sr=16000)
        with torch.no_grad():
            deja[chemin] = model.encode_batch(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
        if n % 100 == 0:
            print(f"  {n}/{len(a_faire)}")
    if a_faire:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, paths=np.array(list(deja.keys())), emb=np.stack(list(deja.values())))
        print(f"Calcul terminé en {time.time() - debut:.0f} s")
    return deja


def scores_cosinus(paires, emb):
    """Cosinus entre les deux empreintes de chaque paire."""
    A = np.stack([emb[f] for f in paires["file1"]])
    B = np.stack([emb[f] for f in paires["file2"]])
    A /= np.linalg.norm(A, axis=1, keepdims=True)   # longueur 1 : seul l'angle compte
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return (A * B).sum(axis=1)


if __name__ == "__main__":
    # 1. Tous les fichiers des paires de validation (et de test, pour plus tard)
    tables = [pd.read_csv(f) for f in FICHIERS_PAIRES if Path(f).exists()]
    chemins = sorted(set(pd.concat(tables)[["file1", "file2"]].values.ravel()))

    model = charger_modele()
    emb = calculer_empreintes(chemins, model)

    # 2. Évaluation sur la VALIDATION uniquement (le test reste fermé jusqu'au jour 9)
    p_val = pd.read_csv("data/pairs/pairs_val.csv")
    s_val = scores_cosinus(p_val, emb)
    r = evaluate(s_val, p_val["label"], nom="ecapa_cosinus")
    enregistrer("ECAPA (speechbrain 1.1.2) + cosinus", "val", r,
                notes="aucun entraînement, réglages par défaut")

    print(f"\nECAPA cosinus | EER val {r['eer']:.3f} | minDCF {r['min_dcf']:.3f} "
          f"| AUC {r['auc']:.3f} | seuil EER {r['seuil_eer']:.3f}")
    print(f"Score moyen même personne      : {s_val[p_val['label'] == 1].mean():.3f}")
    print(f"Score moyen personnes différ.  : {s_val[p_val['label'] == 0].mean():.3f}")

    # 3. Tableau comparatif complet (baseline + ECAPA)
    print("\n", pd.read_csv("results/resultats.csv")[["approche", "paires", "eer", "min_dcf", "auc"]]
          .round(3).to_string(index=False))