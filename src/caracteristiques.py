# src/caracteristiques.py
# Caractéristiques vocales EXPLICABLES (qu'on peut nommer), mesurées sur un signal audio.
# Servent à l'analyse "qu'est-ce qui différencie deux voix" et au panneau d'explication de la démo.
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

SR = 16000

# nom de colonne -> (nom lisible, unité, ce que ça décrit)
DESCRIPTION = {
    "f0_hz":        ("Hauteur moyenne (F0)", "Hz", "grave ou aiguë"),
    "f0_var_dt":    ("Variation de hauteur", "demi-tons", "voix monotone ou chantante"),
    "f1_hz":        ("Formant F1", "Hz", "ouverture de la bouche"),
    "f2_hz":        ("Formant F2", "Hz", "position de la langue (avant / arrière)"),
    "f3_hz":        ("Formant F3", "Hz", "longueur et forme du conduit vocal"),
    "jitter_pct":   ("Jitter", "%", "petites irrégularités de hauteur"),
    "shimmer_pct":  ("Shimmer", "%", "petites irrégularités de volume"),
    "hnr_db":       ("HNR", "dB", "voix claire ou voilée (souffle)"),
    "centroide_hz": ("Centroïde spectral", "Hz", "brillance du timbre"),
    "rythme_s":     ("Rythme (approx.)", "attaques/s", "vitesse de parole"),
    **{f"mfcc{i}": (f"MFCC {i}", "", "forme du spectre (timbre)") for i in range(1, 13)},
}
COLONNES = list(DESCRIPTION)


def extraire(y, sr=SR):
    """Toutes les caractéristiques d'un signal (parties voisées pour la hauteur et les formants)."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    res = {}

    # Hauteur : seulement quand les cordes vocales vibrent
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array["frequency"]
    temps = pitch.xs()[f0 > 0]
    f0 = f0[f0 > 0]
    res["f0_hz"] = float(np.median(f0)) if len(f0) else np.nan
    res["f0_var_dt"] = float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 1 else np.nan

    # Formants : mesurés aux mêmes instants voisés
    formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
    for i in (1, 2, 3):
        v = np.array([formants.get_value_at_time(i, t) for t in temps])
        v = v[np.isfinite(v)]
        res[f"f{i}_hz"] = float(np.median(v)) if len(v) else np.nan

    # Stabilité de la voix
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 60, 400)
        res["jitter_pct"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        res["shimmer_pct"] = 100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0)
        res["hnr_db"] = call(harm, "Get mean", 0, 0)
    except Exception:
        res["jitter_pct"] = res["shimmer_pct"] = res["hnr_db"] = np.nan

    # Timbre et rythme (librosa), sur les trames où il y a du son
    rms = librosa.feature.rms(y=y)[0]
    actif = rms > 0.1 * rms.max()
    res["centroide_hz"] = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0][actif].mean())
    res["rythme_s"] = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)[:, actif]
    for i in range(1, 13):                       # le MFCC 0 (volume) est volontairement exclu
        res[f"mfcc{i}"] = float(mfcc[i].mean())

    return {k: (float(v) if v is not None else np.nan) for k, v in res.items()}


def expliquer(a, b, ecart_intra, n=3):
    """Les n caractéristiques qui séparent le plus deux voix, en unités de 'variation habituelle chez une même personne'.
    a, b : dictionnaires de caractéristiques ; ecart_intra : écart-type moyen chez une même personne, par caractéristique."""
    ecarts = []
    for c in COLONNES:
        if c.startswith("mfcc") or not np.isfinite(a.get(c, np.nan)) or not np.isfinite(b.get(c, np.nan)):
            continue
        z = (b[c] - a[c]) / ecart_intra[c]
        nom, unite, sens = DESCRIPTION[c]
        ecarts.append((abs(z), f"{nom} : {a[c]:.1f} → {b[c]:.1f} {unite} ({sens})"))
    return [texte for _, texte in sorted(ecarts, reverse=True)[:n]]