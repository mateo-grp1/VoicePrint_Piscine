# src/convert_audio.py
# Convertit les MP3 de metadata.csv en wav mono 16 kHz, mesure les durées,
# retire les clips trop courts, puis met à jour data/metadata.csv.
# À lancer depuis la racine du projet : python src/convert_audio.py
import subprocess
import pandas as pd
import soundfile as sf
from pathlib import Path

META = Path("data/metadata.csv")
WAV_DIR = Path("data/processed/cv26_fr_wav")   # les wav vont ici (toujours dans data/, donc hors GitHub)
DUREE_MIN = 2.0                                # secondes : on retire les clips plus courts
MIN_CLIPS = 5                                  # on revérifie qu'un locuteur a encore au moins 5 clips

meta = pd.read_csv(META)
WAV_DIR.mkdir(parents=True, exist_ok=True)

wav_paths, durees, erreurs = [], [], 0
for i, mp3 in enumerate(meta["file_path"]):
    wav = WAV_DIR / (Path(mp3).stem + ".wav")
    if not wav.exists():                       # si déjà converti, on saute (on peut relancer sans tout refaire)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav.as_posix()])
        if r.returncode != 0:
            erreurs += 1
    try:
        durees.append(sf.info(wav).duration)   # durée en secondes, lue dans l'en-tête du wav
    except Exception:
        durees.append(0.0)                     # fichier absent ou cassé -> durée 0, il sera retiré
    wav_paths.append(wav.as_posix())
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{len(meta)} fichiers traités")

meta["wav_path"] = wav_paths
meta["duration_s"] = durees
avant = len(meta)

# Retirer les clips trop courts, puis les locuteurs qui n'ont plus assez de clips
meta = meta[meta["duration_s"] >= DUREE_MIN]
n = meta.groupby("speaker_id")["wav_path"].transform("size")
meta = meta[n >= MIN_CLIPS].reset_index(drop=True)

print("Erreurs de conversion :", erreurs)
print("Clips retirés :", avant - len(meta), "| Clips gardés :", len(meta))
print("Locuteurs gardés :", meta["speaker_id"].nunique())
print("Durée totale :", round(meta["duration_s"].sum() / 3600, 2), "h")
print(meta["duration_s"].describe().round(2))
print("Locuteurs par genre :")
print(meta.groupby("gender", dropna=False)["speaker_id"].nunique())

meta.to_csv(META, index=False)
print("Mis à jour :", META)