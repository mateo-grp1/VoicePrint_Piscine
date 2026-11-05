# src/features_mfcc.py
# Calcule un vecteur de 40 nombres (moyenne + écart-type de 20 MFCC) pour chaque clip.
# À lancer depuis la racine : python src/features_mfcc.py
import numpy as np
import pandas as pd
import librosa
from pathlib import Path

META = Path("data/metadata.csv")
OUT = Path("data/features")
N_MFCC, N_FFT, HOP = 20, 512, 160          # mêmes réglages que dans le notebook

meta = pd.read_csv(META)
OUT.mkdir(exist_ok=True)
X = np.zeros((len(meta), 2 * N_MFCC), dtype=np.float32)

for i, chemin in enumerate(meta["wav_path"]):
    y, sr = librosa.load(chemin, sr=16000)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    X[i] = np.concatenate([m.mean(axis=1), m.std(axis=1)])
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{len(meta)} clips traités")

np.savez(OUT / "mfcc40.npz", X=X, paths=np.array(meta["wav_path"].tolist(), dtype=str))
print("Sauvegardé :", OUT / "mfcc40.npz", "| forme :", X.shape)