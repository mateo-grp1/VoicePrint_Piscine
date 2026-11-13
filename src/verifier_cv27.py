"""
Vérifie le corpus Common Voice 27.0 préparé sur Colab, AVANT d'entraîner le metric learning v2.

1) Anti-fuite : aucun locuteur de validation ni de TEST cv26 ne doit être dans l'entraînement cv27.
   On compare par client_id ET par noms de clips, split par split.
2) Cohérence Colab <-> PC : l'empreinte ECAPA d'un même signal synthétique doit être
   quasi identique sur Colab (GPU) et sur ton PC (CPU). Sinon les empreintes ne seraient pas comparables.

Fichiers attendus (téléchargés depuis Mon Drive/voiceprint) dans data/cv27/ :
    emb_cv27_train.npz, selection_cv27.csv, controle_ecapa.npy

Lancer depuis la racine :  python src/verifier_cv27.py
Sortie : data/cv27/locuteurs_a_retirer.csv (locuteurs cv27 à écarter de l'entraînement, s'il y en a)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exporter_exclusions import METADATA, RACINE, lire_tsv_bruts, nom_clip  # noqa: E402

DOSSIER = RACINE / "data" / "cv27"


def table_cv26():
    """Une ligne par clip cv26 : clip, client_id (si connu), split."""
    meta = pd.read_csv(METADATA, dtype=str)
    meta["clip"] = meta["wav_path"].map(nom_clip)
    meta["split"] = meta["split"].str.lower()
    bruts = lire_tsv_bruts()
    if bruts is not None:
        meta = meta.merge(bruts.drop_duplicates("clip"), on="clip", how="left")
    else:
        meta["client_id"] = np.nan
    if meta["client_id"].isna().all() and meta["speaker_id"].str.len().min() >= 40:
        meta["client_id"] = meta["speaker_id"]          # speaker_id est déjà un client_id
    return meta


def verifier_fuite():
    sel = pd.read_csv(DOSSIER / "selection_cv27.csv", dtype=str)
    cv26 = table_cv26()
    ids_sel, clips_sel = set(sel["client_id"]), set(sel["clip"])

    print("=== 1) Anti-fuite cv26 -> entraînement cv27 ===")
    print(f"Entraînement cv27 : {sel['locuteur'].nunique()} locuteurs, {len(sel)} clips\n")
    print(f"{'split cv26':<12}{'locuteurs':>10}{'clips':>8}{'communs (client_id)':>22}{'communs (clips)':>17}")
    a_retirer = set()
    for split, g in cv26.groupby("split"):
        ids = set(g["client_id"].dropna())
        communs_ids = ids & ids_sel
        communs_clips = set(g["clip"]) & clips_sel
        loc_par_clips = set(sel.loc[sel["clip"].isin(communs_clips), "client_id"])
        print(f"{split:<12}{g['speaker_id'].nunique():>10}{len(g):>8}{len(communs_ids):>22}{len(communs_clips):>17}")
        if split != "train":
            a_retirer |= communs_ids | loc_par_clips

    connus = cv26["client_id"].notna().mean()
    print(f"\nclip cv26 dont le client_id est connu : {connus:.0%}")
    if connus < 1:
        print("ATTENTION : client_id inconnu pour une partie des clips -> vérification par noms de clips seulement pour eux.")

    retirer = sel[sel["client_id"].isin(a_retirer)].drop_duplicates("locuteur")[["locuteur", "client_id"]]
    retirer.to_csv(DOSSIER / "locuteurs_a_retirer.csv", index=False)
    if len(retirer):
        print(f"\n⚠️  {len(retirer)} locuteur(s) de validation/test cv26 présents dans l'entraînement cv27 -> "
              f"listés dans data/cv27/locuteurs_a_retirer.csv (ils seront exclus par le metric learning v2).")
    else:
        print("\n✅ Aucun locuteur de validation ni de test cv26 dans l'entraînement cv27.")


def verifier_controle():
    print("\n=== 2) Cohérence des empreintes Colab <-> PC ===")
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    try:
        from speechbrain.utils.fetching import LocalStrategy
        options = {"local_strategy": LocalStrategy.COPY}
    except ImportError:
        options = {}
    modele = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            savedir=str(RACINE / "models" / "ecapa"), **options)
    modele.eval()
    L = 48000
    g = np.random.default_rng(0)
    t = np.arange(L) / 16000
    signal = (0.05 * g.standard_normal(L) + 0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    with torch.no_grad():
        pc = modele.encode_batch(torch.from_numpy(signal)[None], torch.ones(1)).squeeze().numpy()
    colab = np.load(DOSSIER / "controle_ecapa.npy")
    cos = float(pc @ colab / (np.linalg.norm(pc) * np.linalg.norm(colab)))
    print(f"Cosinus entre l'empreinte Colab et l'empreinte PC : {cos:.5f}")
    print("✅ Cohérent (> 0,999)" if cos > 0.999 else "⚠️  Écart anormal : envoie-moi ce résultat.")


def resume_empreintes():
    print("\n=== 3) Empreintes ===")
    d = np.load(DOSSIER / "emb_cv27_train.npz", allow_pickle=False)
    emb = d["emb"]
    print(f"Forme : {emb.shape} | locuteurs : {len(np.unique(d['locuteur']))} | versions : {d['noms_versions'].tolist()}")
    print(f"NaN : {int(np.isnan(emb).sum())} | norme moyenne : {np.linalg.norm(emb, axis=1).mean():.1f}")


if __name__ == "__main__":
    for f in ("emb_cv27_train.npz", "selection_cv27.csv", "controle_ecapa.npy"):
        if not (DOSSIER / f).exists():
            sys.exit(f"ERREUR : {DOSSIER / f} introuvable. Télécharge-le depuis Mon Drive/voiceprint.")
    verifier_fuite()
    resume_empreintes()
    verifier_controle()