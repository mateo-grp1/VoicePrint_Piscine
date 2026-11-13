# src/augmentation.py
# Outils partagés : bruits réalistes, mélange à un niveau voulu, empreintes ECAPA calculées par lots.
import numpy as np
import torch

SR = 16000


def bruit_rose(n, rng):
    """Bruit de fond continu, plus fort dans les graves (ventilation, salle)."""
    spectre = np.fft.rfft(rng.normal(size=n))
    f = np.arange(len(spectre))
    spectre[1:] /= np.sqrt(f[1:])
    b = np.fft.irfft(spectre, n)
    return b / (np.abs(b).max() + 1e-9)


def brouhaha(n, voix, rng, nb=4):
    """Plusieurs voix superposées à des moments différents (des gens qui discutent autour)."""
    b = np.zeros(n)
    for v in rng.choice(len(voix), nb, replace=False):
        y = voix[v]
        debut = rng.integers(0, max(1, len(y) - n))
        morceau = y[debut:debut + n]
        b[:len(morceau)] += morceau / (np.std(morceau) + 1e-9)
    return b


def melanger(y, b, snr_db):
    """Ajoute le bruit b au signal y avec le rapport signal/bruit voulu (en dB)."""
    p_signal, p_bruit = np.mean(y ** 2), np.mean(b ** 2) + 1e-12
    return (y + b * np.sqrt(p_signal / (p_bruit * 10 ** (snr_db / 10)))).astype(np.float32)


def empreintes_signaux(model, signaux, lot=32):
    """Empreintes ECAPA (192 nombres) de plusieurs signaux, calculées par lots (plus rapide sur CPU)."""
    out = []
    for i in range(0, len(signaux), lot):
        paquet = signaux[i:i + lot]
        n = max(len(y) for y in paquet)
        x = np.zeros((len(paquet), n), dtype=np.float32)
        for j, y in enumerate(paquet):
            x[j, :len(y)] = y
        longueurs = torch.tensor([len(y) / n for y in paquet], dtype=torch.float32)
        with torch.no_grad():
            e = model.encode_batch(torch.tensor(x), longueurs)
        out.append(np.asarray(e.reshape(len(paquet), -1)))
    return np.vstack(out)