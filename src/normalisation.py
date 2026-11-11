# src/normalisation.py
# Outils de normalisation des empreintes et des scores, partagés par les scripts du jour 3 et la future démo.
import numpy as np
import pandas as pd

GRAINE = 42
CLIPS_TRAIN_PAR_LOC = 10     # suffisant pour une moyenne et une cohorte, et rapide à calculer


def clips_train(n_par_loc=CLIPS_TRAIN_PAR_LOC):
    """Quelques clips par locuteur d'ENTRAÎNEMENT (jamais de validation ni de test)."""
    meta = pd.read_csv("data/metadata.csv")
    train = meta[meta["split"] == "train"]
    # on mélange une fois (graine fixe), puis on garde les n premiers clips de chaque locuteur
    return train.sample(frac=1, random_state=GRAINE).groupby("speaker_id").head(n_par_loc)


def unitaire(E):
    """Ramène chaque empreinte (ligne) à une longueur de 1."""
    E = np.atleast_2d(E)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def preparer(E, moyenne=None):
    """Longueur 1, puis (option) soustraction de l'empreinte moyenne du train, puis longueur 1 à nouveau."""
    E = unitaire(E)
    if moyenne is not None:
        E = unitaire(E - moyenne)
    return E


def construire_reference(emb, train, moyenne=None):
    """Empreinte moyenne du train (à soustraire) et cohorte : une empreinte moyenne par locuteur de train."""
    E = unitaire(np.stack([emb[p] for p in train["wav_path"]]))
    mu = E.mean(axis=0)
    cohorte = []
    for _, g in train.groupby("speaker_id"):
        Eg = preparer(np.stack([emb[p] for p in g["wav_path"]]), mu if moyenne else None)
        cohorte.append(unitaire(Eg.mean(axis=0))[0])
    return mu, np.stack(cohorte)


def as_norm(scores, A, B, cohorte, top_k=30):
    """AS-norm : chaque score est comparé aux scores habituels de A et de B contre des imposteurs (la cohorte).
    Score normalisé = moyenne de (s - moy_A) / écart_A et (s - moy_B) / écart_B, sur les top_k imposteurs les plus proches."""
    def stats(X):
        s = np.sort(X @ cohorte.T, axis=1)[:, -top_k:]
        return s.mean(axis=1), s.std(axis=1) + 1e-8
    ma, sa = stats(A)
    mb, sb = stats(B)
    return 0.5 * ((scores - ma) / sa + (scores - mb) / sb)