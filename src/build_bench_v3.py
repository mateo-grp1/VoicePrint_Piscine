# src/build_bench_v3.py
# Banc d'essai v3 : les MÊMES paires et les MÊMES personnes que le banc v2,
# mais chaque voix est préparée EXACTEMENT comme dans la démo (src/preparation.py).
#
# Pourquoi : le banc v2 prenait "le milieu du clip brut" (silences compris), alors que la démo
# retire tous les silences puis garde les N premières secondes de PAROLE. Seuil et calibration
# doivent être mesurés dans les conditions du jour J.
#
# Contenu :
#   - enrôlement : 3, 5 ou 10 s de parole de la personne enrôlée (un bloc : clip file1, puis si besoin
#     ses autres clips, jamais le clip de test) ;
#   - test : "3s" (3 premières secondes de parole) et "max10s" (toute la parole jusqu'à 10 s, comme la démo) ;
#   - conditions du test : propre, ambiance 10 dB, brouhaha 10 dB. Le bruit est ajouté à
#     l'enregistrement BRUT, AVANT le retrait des silences, comme dans une vraie salle.
# Les empreintes BRUTES (sans moyenne soustraite) sont mises en cache : data/embeddings/banc_v3.npz
# À lancer depuis la racine : python src/build_bench_v3.py   (environ 15-30 min sur CPU la 1re fois)
import time
from pathlib import Path

import numpy as np
import pandas as pd

from augmentation import brouhaha, bruit_rose, melanger
from normalisation import clips_train
from preparation import SR, charger_brut, empreinte_brute, enrolement, garder_parole, test

GRAINE = 42
DUREES_ENROL = [3, 5, 10]
TESTS = {"3s": 3, "max10s": 10}
CONDITIONS = {"propre": None, "ambiance": 10, "brouhaha": 10}
BANC = Path("data/pairs/pairs_val_v3.csv")
CACHE = Path("data/embeddings/banc_v3.npz")


def construire_paires():
    """Mêmes paires que le banc v2 (version "entier" = fichiers d'origine)."""
    v2 = pd.read_csv("data/pairs/pairs_val_v2.csv")
    p = v2[v2["duree"] == "entier"].drop(columns="duree").reset_index(drop=True)
    p["exclu"] = np.where(p["label"] == 1, p["file2"], "")   # le clip de test n'entre jamais dans l'enrôlement
    return p


def cle_enrol(d, file1, exclu):
    return f"enrol|{d}|{file1}|{exclu}"


def cle_test(mode, cond, file2):
    return f"test|{mode}|{cond}|{file2}"


if __name__ == "__main__":
    rng = np.random.default_rng(GRAINE)
    paires = construire_paires()
    BANC.parent.mkdir(parents=True, exist_ok=True)
    paires.to_csv(BANC, index=False)
    print(f"Paires : {len(paires)} ({paires['type'].value_counts().to_dict()})")

    # Clips de chaque personne de validation (dans l'ordre, pour un enrôlement reproductible)
    v2 = pd.read_csv("data/pairs/pairs_val_v2.csv")
    v2 = v2[v2["duree"] == "entier"]
    clips = pd.concat([v2[["file1", "spk1"]].set_axis(["f", "s"], axis=1),
                       v2[["file2", "spk2"]].set_axis(["f", "s"], axis=1)]).drop_duplicates("f")
    par_loc = clips.groupby("s")["f"].apply(sorted).to_dict()

    parole_cache = {}

    def parole(f):
        if f not in parole_cache:
            parole_cache[f] = garder_parole(charger_brut(f))
        return parole_cache[f]

    # ---------- Signaux à calculer ----------
    a_faire = {}
    for f1, s1, exclu in paires[["file1", "spk1", "exclu"]].drop_duplicates().itertuples(index=False):
        autres = [f for f in par_loc[s1] if f not in (f1, exclu)]
        bloc = np.concatenate([parole(f1)] + [parole(f) for f in autres])   # file1 d'abord, puis les autres
        for d in DUREES_ENROL:
            a_faire[cle_enrol(d, f1, exclu)] = enrolement(bloc, d)

    voix = [garder_parole(charger_brut(f)) for f in clips_train()["wav_path"].sample(60, random_state=GRAINE + 1)]
    for f2 in sorted(set(paires["file2"])):
        brut = charger_brut(f2)
        for cond, snr in CONDITIONS.items():
            if snr is None:
                b = brut
            else:
                bruit = bruit_rose(len(brut), rng) if cond == "ambiance" else brouhaha(len(brut), voix, rng)
                b = melanger(brut, bruit, snr)
            p = garder_parole(b)                                  # la recette de la démo, APRÈS le bruit
            for mode, duree in TESTS.items():
                a_faire[cle_test(mode, cond, f2)] = test(p, duree)

    # Clips de train cv26 préparés comme la démo : pour une moyenne soustraite "même recette"
    for f in clips_train()["wav_path"]:
        a_faire[f"train|{f}"] = test(garder_parole(charger_brut(f)))

    courts = sum(1 for k, y in a_faire.items() if k.startswith("enrol|") and len(y) < int(k.split("|")[1]) * SR)
    print(f"Signaux : {len(a_faire)} (enrôlements plus courts que prévu, faute de parole : {courts})")

    # ---------- Empreintes (cache : on ne recalcule que ce qui manque) ----------
    deja = {}
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as d:
            deja = dict(zip(d["cles"], d["emb"]))
    manque = [k for k in a_faire if k not in deja]
    print(f"{len(deja)} empreintes en cache, {len(manque)} à calculer")
    if manque:
        from ecapa_eval import charger_modele
        modele = charger_modele()
        debut = time.time()
        for n, k in enumerate(manque, 1):
            deja[k] = empreinte_brute(modele, a_faire[k])
            if n % 250 == 0:
                reste = (time.time() - debut) / n * (len(manque) - n)
                print(f"  {n}/{len(manque)}  (encore environ {reste / 60:.0f} min)")
                np.savez(CACHE, cles=np.array(list(deja)), emb=np.stack(list(deja.values())))   # sauvegarde d'étape
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, cles=np.array(list(deja)), emb=np.stack(list(deja.values())))
    print(f"Banc v3 prêt : {BANC} | empreintes : {CACHE}")