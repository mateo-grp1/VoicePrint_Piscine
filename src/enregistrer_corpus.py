# src/enregistrer_corpus.py
# Jour 4 : enregistrer le mini-corpus "vraie vie" (toi + quelques proches volontaires).
# Même micro, même format pour tout le monde : wav mono 16 kHz, rangé dans data/mini_corpus/ (exclu de GitHub).
# Les personnes sont désignées par un code (p00, p01...), jamais par leur nom.
# Installation (une fois) : pip install sounddevice
# À lancer depuis la racine : python src/enregistrer_corpus.py
import time
from datetime import date
from pathlib import Path
import pandas as pd
import sounddevice as sd
import soundfile as sf

SR = 16000
DOSSIER = Path("data/mini_corpus")
CONSENTEMENT = (
    "Projet Voiceprint (piscine IA Lab). Tes enregistrements servent uniquement à tester un système\n"
    "de reconnaissance de voix. Ils restent sur cet ordinateur, ne sont jamais publiés ni partagés,\n"
    "et sont supprimés à la fin du projet ou dès que tu le demandes."
)


def enregistrer(duree, consigne):
    """Compte à rebours, enregistre, puis propose de réécouter et de recommencer."""
    while True:
        print(f"\n>>> {consigne}  ({duree} s)")
        input("    Appuie sur Entrée quand tu es prêt(e)...")
        for i in (3, 2, 1):
            print(f"    {i}...", end=" ", flush=True)
            time.sleep(1)
        print("PARLE !")
        y = sd.rec(int(duree * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        y = y[:, 0]
        print(f"    Terminé. Volume max : {abs(y).max():.2f} (idéal entre 0.2 et 0.9)")
        choix = input("    [Entrée] garder · [e] écouter · [r] recommencer : ").strip().lower()
        if choix == "e":
            sd.play(y, SR)
            sd.wait()
            choix = input("    [Entrée] garder · [r] recommencer : ").strip().lower()
        if choix != "r":
            return y


print(CONSENTEMENT)
if input("\nLa personne est-elle d'accord ? (oui/non) : ").strip().lower() != "oui":
    raise SystemExit("Pas de consentement : rien n'est enregistré.")

code = input("Code de la personne (p00 = toi, p01, p02...) : ").strip()
genre = input("Genre (h / f / autre / non précisé) : ").strip()
langues = [l.strip() for l in input("Langues parlées, la principale d'abord (ex. fr,es,en) : ").split(",") if l.strip()]

# Le protocole : un enrôlement de 10 s (on en découpera 3, 5 et 8 s), puis des tests de 5 s (on en gardera 3 s)
protocole = [("enrolement", langues[0], 10, f"ENRÔLEMENT en {langues[0]} : parle librement (présente-toi, raconte ta journée)")]
for n in (1, 2):
    protocole.append((f"test_{langues[0]}_{n}", langues[0], 5, f"TEST en {langues[0]} : une autre phrase, différente de l'enrôlement"))
for langue in langues[1:3]:
    for n in (1, 2):
        protocole.append((f"test_{langue}_{n}", langue, 5, f"TEST en {langue} : parle librement"))
protocole.append(("test_fausse_langue", "aucune", 5, "TEST : invente une fausse langue (des sons, pas de vrais mots)"))
protocole.append(("test_loin", langues[0], 5, f"TEST en {langues[0]} : recule à environ 1,5 m du micro"))

dossier = DOSSIER / code
dossier.mkdir(parents=True, exist_ok=True)
lignes = []
for condition, langue, duree, consigne in protocole:
    y = enregistrer(duree, consigne)
    fichier = (dossier / f"{condition}.wav").as_posix()
    sf.write(fichier, y, SR)
    lignes.append({"personne": code, "genre": genre, "fichier": fichier, "condition": condition,
                   "langue": langue, "micro": "pc", "date": date.today().isoformat()})

meta = DOSSIER / "metadata.csv"
nouveau = pd.DataFrame(lignes)
if meta.exists():
    ancien = pd.read_csv(meta)
    nouveau = pd.concat([ancien[ancien["personne"] != code], nouveau], ignore_index=True)
nouveau.to_csv(meta, index=False)
print(f"\n{len(lignes)} enregistrements sauvegardés dans {dossier}. Merci !")