# src/test_final.py
# TEST FINAL : la configuration figée de la démo, évaluée UNE SEULE FOIS sur les 29 personnes de TEST cv26,
# jamais utilisées jusqu'ici (aucun réglage, aucun choix n'a été fait avec elles).
#
# Protocole (comme la démo et la grande validation cv27) :
#   - chaque personne s'enrôle une fois : 3, 5 ou 10 s de parole (un bloc fait de ses clips d'enrôlement) ;
#   - jusqu'à 10 de ses autres clips servent de tests ; TOUS les tests passent devant TOUS les enrôlements ;
#   - test de 3 s ou jusqu'à 10 s ; propre, ambiance 10 dB, brouhaha 10 dB (bruit ajouté avant le retrait des silences) ;
#   - voix préparées par src/preparation.py, exactement comme l'application ;
#   - imposteurs "voix proches" = le quart des couples de personnes dont la hauteur de voix (Praat) est la plus proche.
# Mesures : EER, minDCF, AUC, ROC, et surtout les taux d'erreur AU SEUIL FIGÉ de la démo.
#
# Rien n'est réglé ici : le script lit models/reference_demo.npz et ne le modifie pas.
# À lancer depuis la racine : python src/test_final.py   (environ 10 à 20 min sur CPU, cache ensuite)
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth

from augmentation import brouhaha, bruit_rose, melanger
from evaluate import enregistrer, evaluate
from normalisation import clips_train, preparer
from preparation import SR, charger_brut, empreinte_brute, enrolement, garder_parole, test

GRAINE = 42
N_TESTS_MAX = 10           # clips de test par personne au maximum
N_ENROL_MIN = 2            # clips gardés au minimum pour l'enrôlement
DUREES_ENROL = [3, 5, 10]
TESTS = {"3s": 3, "max10s": 10}
CONDITIONS = {"propre": None, "ambiance": 10, "brouhaha": 10}
CACHE = Path("data/embeddings/test_final.npz")
rng = np.random.default_rng(GRAINE)

# ---------- Configuration figée (lecture seule) ----------
with np.load("models/reference_demo.npz") as r:
    MU = r["moyenne"].copy()
    SEUILS = {int(k): float(v) for k, v in json.loads(str(r["seuils"])).items()}
    CAL_A, CAL_B = float(r["calib_a"]), float(r["calib_b"])
print("Configuration figée : seuils " + " | ".join(f"{d} s : {s:.3f}" for d, s in SEUILS.items()))

# ---------- Personnes de TEST et répartition enrôlement / tests ----------
meta = pd.read_csv("data/metadata.csv")
tst = meta[meta["split"] == "test"]
locs = sorted(tst["speaker_id"].unique())
assert set(locs).isdisjoint(set(meta.loc[meta["split"] != "test", "speaker_id"])), "FUITE : locuteur de test ailleurs !"
plan_enrol, plan_tests = {}, []
for s in locs:
    fichiers = list(rng.permutation(sorted(tst.loc[tst["speaker_id"] == s, "wav_path"])))
    n_tests = min(N_TESTS_MAX, len(fichiers) - N_ENROL_MIN)
    plan_tests += [(f, s) for f in fichiers[:n_tests]]
    plan_enrol[s] = fichiers[n_tests:]
print(f"Test : {len(locs)} personnes, {len(plan_tests)} clips de test")

# ---------- Hauteur de voix (pour les voix proches) ----------
def f0_mediane(fichiers):
    v = []
    for f in fichiers[:5]:
        p = parselmouth.Sound(garder_parole(charger_brut(f)).astype(np.float64), SR).to_pitch(
            time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]
        v.extend(p[p > 0])
    return float(np.median(v))


f0 = {s: f0_mediane(plan_enrol[s] + [f for f, t in plan_tests if t == s]) for s in locs}
couples = sorted(combinations(locs, 2), key=lambda c: abs(f0[c[0]] - f0[c[1]]))
proches = {frozenset(c) for c in couples[:len(couples) // 4]}

# ---------- Empreintes (cache) ----------
cles_attendues = [f"enrol|{d}|{s}" for s in locs for d in DUREES_ENROL] + \
                 [f"test|{m}|{c}|{f}" for f, _ in plan_tests for c in CONDITIONS for m in TESTS]
emb = {}
if CACHE.exists():
    with np.load(CACHE, allow_pickle=False) as d:
        emb = dict(zip(d["cles"], d["emb"]))
if not all(k in emb for k in cles_attendues):
    from ecapa_eval import charger_modele
    modele = charger_modele()
    voix = [garder_parole(charger_brut(f)) for f in clips_train()["wav_path"].sample(60, random_state=GRAINE + 1)]
    debut, courts = time.time(), 0
    for s in locs:
        bloc = np.concatenate([garder_parole(charger_brut(f)) for f in plan_enrol[s]])
        for d in DUREES_ENROL:
            y = enrolement(bloc, d)
            courts += len(y) < d * SR
            emb[f"enrol|{d}|{s}"] = empreinte_brute(modele, y)
    print(f"Enrôlements calculés ({courts} plus courts que prévu, faute de parole)")
    for n, (f, _) in enumerate(plan_tests, 1):
        brut = charger_brut(f)
        for c, snr in CONDITIONS.items():
            if snr is None:
                b = brut
            else:
                bruit = bruit_rose(len(brut), rng) if c == "ambiance" else brouhaha(len(brut), voix, rng)
                b = melanger(brut, bruit, snr)
            p = garder_parole(b)
            for m, duree in TESTS.items():
                emb[f"test|{m}|{c}|{f}"] = empreinte_brute(modele, test(p, duree))
        if n % 50 == 0:
            print(f"  {n}/{len(plan_tests)} clips de test ({time.time() - debut:.0f} s)")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, cles=np.array(list(emb)), emb=np.stack(list(emb.values())))

# ---------- Essais : chaque enrôlement contre chaque clip de test ----------
spk_test = np.array([s for _, s in plan_tests])
spk_enrol = np.array(locs)
LABEL = spk_enrol[:, None] == spk_test[None, :]
PROCHE = np.array([[frozenset((a, b)) in proches for b in spk_test] for a in spk_enrol])
DIFF = LABEL | PROCHE
print(f"Essais : {int(LABEL.sum())} de la vraie personne | {int((~LABEL).sum())} d'imposteurs "
      f"dont {int((PROCHE & ~LABEL).sum())} voix proches")


def scores(d, m, c):
    E = preparer(np.stack([emb[f"enrol|{d}|{s}"] for s in locs]), MU)
    T = preparer(np.stack([emb[f"test|{m}|{c}|{f}"] for f, _ in plan_tests]), MU)
    return E @ T.T


lignes = []
for d in DUREES_ENROL:
    for m in TESTS:
        for c in CONDITIONS:
            s = scores(d, m, c)
            for cas, sel in [("voix proches", DIFF), ("tous imposteurs", np.ones_like(DIFF))]:
                nom = f"test_final_{d}s_{m}_{c}" if (d == 3 and m == "max10s" and cas == "voix proches") else None
                r = evaluate(s[sel], LABEL[sel], nom=nom)
                seuil = SEUILS[d]
                fa = (s[sel & ~LABEL] >= seuil).mean()
                frr = (s[LABEL] < seuil).mean()
                enregistrer(f"CONFIG FINALE | TEST cv26 (29 pers.) | enrôlement {d} s | test {m} | {c} | {cas}",
                            "test", {**r, "fa_seuil_demo": fa, "frr_seuil_demo": frr},
                            notes="évaluation unique, configuration figée")
                lignes.append({"enrôlement": f"{d} s", "test": m, "condition": c, "imposteurs": cas,
                               "EER": r["eer"], "minDCF": r["min_dcf"], "AUC": r["auc"],
                               "seuil démo": seuil, "imposteurs acceptés (seuil démo)": fa,
                               "vraies personnes refusées (seuil démo)": frr})
res = pd.DataFrame(lignes)
res.to_csv("results/test_final.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== TEST FINAL, voix proches ===")
print(res[res["imposteurs"] == "voix proches"].drop(columns="imposteurs").round(3).to_string(index=False))
print("\n=== TEST FINAL, tous les imposteurs (hasard + proches), test propre ===")
print(res[(res["imposteurs"] == "tous imposteurs") & (res["condition"] == "propre")]
      .drop(columns=["imposteurs", "condition"]).round(3).to_string(index=False))

# Calibration sur le test, essais équilibrés
s = scores(3, "max10s", "propre")
pos, neg = s[LABEL], rng.choice(s[~LABEL], size=int(LABEL.sum()), replace=False)
p = 1 / (1 + np.exp(-(CAL_A * np.r_[pos, neg] + CAL_B)))
cal = pd.DataFrame({"tranche": pd.cut(p, [0, 0.1, 0.5, 0.9, 1.0], include_lowest=True),
                    "label": np.r_[np.ones(len(pos)), np.zeros(len(neg))]})
print("\n=== Calibration de la démo sur le test (essais équilibrés) ===")
print(cal.groupby("tranche", observed=True)["label"].agg(essais="size", vraie_personne_en_realite="mean").round(3).to_string())
print("\nRésultats : results/test_final.csv | ROC : results/roc_test_final_3s_max10s_*.png")