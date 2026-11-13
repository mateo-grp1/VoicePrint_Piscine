"""
Exporte la liste des locuteurs cv26 à EXCLURE de l'entraînement du metric learning v2.

Pourquoi : le nouveau corpus Common Voice 27.0 contient très probablement les mêmes
personnes que notre cv26. Si un locuteur de validation ou de test se retrouvait dans
l'entraînement, nos chiffres seraient faussés (fuite de données).

On exclut TOUT locuteur cv26 qui n'est pas dans le split "train" (validation + test).
On exporte deux types d'identifiants, pour une vérification double sur Colab :
  - client_id : identifiant (haché) du locuteur dans Common Voice
  - clip      : nom du fichier audio sans extension (ex. common_voice_fr_12345678),
                stable d'une version de Common Voice à l'autre

Lancer depuis la racine du projet :
    python src/exporter_exclusions.py

Sortie : data/exclusions_cv26.csv (hors GitHub, car data/ est dans le .gitignore)
À déposer ensuite dans Google Drive : Mon Drive/voiceprint/exclusions_cv26.csv
"""
import csv
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
METADATA = RACINE / "data" / "metadata.csv"
DOSSIER_BRUT = RACINE / "data" / "raw" / "cv26_fr"
SORTIE = RACINE / "data" / "exclusions_cv26.csv"


def nom_clip(chemin: str) -> str:
    """'data/raw/.../common_voice_fr_123.mp3' -> 'common_voice_fr_123'"""
    return Path(str(chemin).replace("\\", "/")).stem


def lire_tsv_bruts():
    """Lit les .tsv d'origine de cv26 (s'ils existent) pour relier clip -> client_id."""
    lignes = []
    for tsv in sorted(DOSSIER_BRUT.rglob("*.tsv")):
        try:
            df = pd.read_csv(tsv, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                             usecols=lambda c: c in ("client_id", "path"),
                             on_bad_lines="skip")
        except Exception:
            continue
        if {"client_id", "path"} <= set(df.columns):
            df["clip"] = df["path"].map(nom_clip)
            lignes.append(df[["client_id", "clip"]])
    if not lignes:
        return None
    return pd.concat(lignes).drop_duplicates()


def main():
    if not METADATA.exists():
        sys.exit(f"ERREUR : {METADATA} introuvable. Lance le script depuis la racine du projet.")

    meta = pd.read_csv(METADATA, dtype=str)
    for col in ("wav_path", "speaker_id", "split"):
        if col not in meta.columns:
            sys.exit(f"ERREUR : colonne '{col}' absente de metadata.csv (colonnes : {list(meta.columns)})")

    print("Valeurs de 'split' dans metadata.csv :")
    print(meta["split"].value_counts().to_string(), "\n")

    a_exclure = meta[meta["split"].str.lower() != "train"].copy()
    a_exclure["clip"] = a_exclure["wav_path"].map(nom_clip)
    clips = set(a_exclure["clip"])
    locuteurs = set(a_exclure["speaker_id"])

    # Les speaker_id de metadata.csv sont-ils des client_id Common Voice ?
    # (un client_id est un long hachage hexadécimal, 64 ou 128 caractères)
    ressemble_client_id = all(len(s) >= 40 for s in locuteurs)

    client_ids = set()
    if ressemble_client_id:
        client_ids |= locuteurs
    bruts = lire_tsv_bruts()
    if bruts is not None:
        client_ids |= set(bruts.loc[bruts["clip"].isin(clips), "client_id"])

    lignes = [("client_id", c) for c in sorted(client_ids)] + [("clip", c) for c in sorted(clips)]
    pd.DataFrame(lignes, columns=["type", "valeur"]).to_csv(SORTIE, index=False)

    print(f"Locuteurs cv26 hors train : {len(locuteurs)}")
    print(f"Clips cv26 hors train     : {len(clips)}")
    print(f"client_id exportés        : {len(client_ids)}"
          f"  (speaker_id = client_id ? {'oui' if ressemble_client_id else 'non'} ;"
          f" tsv bruts trouvés ? {'oui' if bruts is not None else 'non'})")
    if not client_ids:
        print("ATTENTION : aucun client_id trouvé. La vérification sur Colab se fera "
              "uniquement par noms de clips (ça reste suffisant si les clips sont retrouvés).")
    print(f"\nFichier écrit : {SORTIE}")
    print("À déposer dans Google Drive : Mon Drive/voiceprint/exclusions_cv26.csv")


if __name__ == "__main__":
    main()