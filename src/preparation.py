# src/preparation.py
# LA recette de préparation de la voix, unique et partagée.
# Utilisée par la démo (app.py) ET par les évaluations (banc v3) : impossible qu'elles divergent.
#
#   1. audio -> 16 kHz mono
#   2. on ne garde QUE la parole : silences du début, de la fin et pauses du milieu retirés (top_db=30),
#      morceaux de parole recollés
#   3. enrôlement = les N premières secondes de parole ; test = toute la parole, jusqu'à 10 s
#   4. empreinte ECAPA du signal (un seul appel, sans remplissage)
import numpy as np
import librosa
import torch

SR = 16000
TOP_DB = 30
TEST_MAX = 10      # secondes de parole analysées au maximum pour un test


def garder_parole(brut):
    """Retire silences et pauses, recolle les morceaux de parole."""
    morceaux = librosa.effects.split(brut, top_db=TOP_DB)
    if len(morceaux) == 0:
        return brut[:0]
    return np.concatenate([brut[a:b] for a, b in morceaux]).astype(np.float32)


def charger_brut(source):
    """Fichier (chemin ou objet fichier) -> signal 16 kHz mono, sans aucun nettoyage."""
    return librosa.load(source, sr=SR, mono=True)[0]


def enrolement(parole, duree):
    """Les `duree` premières secondes de parole."""
    return parole[: int(duree * SR)]


def test(parole, duree_max=TEST_MAX):
    """Toute la parole, jusqu'à `duree_max` secondes."""
    return parole[: int(duree_max * SR)]


def empreinte_brute(modele, y):
    """Empreinte ECAPA (192 nombres) d'un signal, sans normalisation."""
    with torch.no_grad():
        return modele.encode_batch(torch.tensor(y, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()