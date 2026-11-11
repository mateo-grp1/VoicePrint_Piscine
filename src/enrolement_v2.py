# src/enrolement_v2.py
# Jour 3, étape 3 : durée d'enrôlement aux conditions de la démo.
#  - Enrôlement = UN seul enregistrement continu de 3, 5, 8 ou 10 s
#    (on met bout à bout les clips d'une personne, sans leurs silences, puis on coupe à la durée voulue).
#  - Deux façons de calculer l'empreinte de ce MÊME enregistrement :
#      "bloc"     : une seule empreinte sur tout l'enregistrement
#      "morceaux" : moyenne des empreintes de morceaux de 3 s qui se chevauchent
#  - Test = extraits de 3 s (et clips entiers pour comparaison) : ses autres clips (positifs),
#    les clips des autres locuteurs de validation (négatifs).
#  - Les clips de test sont identiques pour toutes les durées : seule la durée d'enrôlement change.
# À lancer depuis la racine : python src/enrolement_v2.py
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
from evaluate import evaluate, enregistrer
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, unitaire, preparer, construire_reference

GRAINE, SR = 42, 16000
DUREES_ENROLEMENT = [3, 5, 8, 10]          # secondes
MORCEAU, PAS = 3.0, 1.5                    # morceaux de 3 s, décalés de 1,5 s
MIN_TEST = 2                               # clips de test minimum par locuteur gardé
rng = np.random.default_rng(GRAINE)


def charger_sans_silence(f):
    y, _ = librosa.load(f, sr=SR)
    return librosa.effects.trim(y, top_db=30)[0]


def empreinte_signal(model, y):
    with torch.no_grad():
        return model.encode_batch(torch.tensor(y).unsqueeze(0)).squeeze().numpy()


def empreinte_morceaux(model, y):
    """Moyenne des empreintes de morceaux de 3 s (un seul morceau si l'audio fait 3 s)."""
    n, h = int(MORCEAU * SR), int(PAS * SR)
    debuts = list(range(0, max(1, len(y) - n + 1), h))
    E = unitaire(np.stack([empreinte_signal(model, y[d:d + n]) for d in debuts]))
    return E.mean(axis=0)


# 1. Clips de validation par locuteur (clips entiers du banc v2), dans un ordre aléatoire fixe
bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
entier = bench[bench["duree"] == "entier"]
clips = pd.concat([entier[["file1", "spk1"]].set_axis(["file", "spk"], axis=1),
                   entier[["file2", "spk2"]].set_axis(["file", "spk"], axis=1)]).drop_duplicates("file")
par_loc = {s: list(rng.permutation(sorted(g["file"]))) for s, g in clips.groupby("spk")}

# 2. Pour chaque locuteur : les premiers clips forment l'enregistrement d'enrôlement (≥ 10 s), les autres servent au test
max_s = max(DUREES_ENROLEMENT)
audio_enrol, tests_loc = {}, {}
for s, fichiers in par_loc.items():
    morceaux, total, i = [], 0.0, 0
    while total < max_s and i < len(fichiers):
        y = charger_sans_silence(fichiers[i])
        morceaux.append(y)
        total += len(y) / SR
        i += 1
    if total >= max_s and len(fichiers) - i >= MIN_TEST:
        audio_enrol[s] = np.concatenate(morceaux)
        tests_loc[s] = fichiers[i:]
print(f"Locuteurs gardés (≥ {max_s} s d'enrôlement et ≥ {MIN_TEST} clips de test) : {len(audio_enrol)} sur {len(par_loc)}")

# Clips de test : ceux des locuteurs gardés (positifs pour eux) + tous les clips des autres (imposteurs seulement)
tests = [(f, s) for s, fs in tests_loc.items() for f in fs]
tests += [(f, s) for s, fs in par_loc.items() if s not in audio_enrol for f in fs]
spk_test = np.array([s for _, s in tests])

# 3. Voix proches : même règle que le banc v2
f0 = pd.read_csv("data/features/f0_locuteurs_val.csv").set_index("spk")
couples = sorted(combinations(f0.index, 2), key=lambda c: abs(f0.f0_hz[c[0]] - f0.f0_hz[c[1]]))
proches = {frozenset(c) for c in couples[:max(1, len(couples) // 4)]}

# 4. Empreintes : tests (cache) + train (moyenne à soustraire) + enrôlements (calculés ici, jamais stockés)
def version(f, duree):
    return f if duree == "entier" else f"data/processed/extraits_{duree}/{Path(f).name}"

model = charger_modele()
train = clips_train()
emb = calculer_empreintes(sorted({version(f, d) for f, _ in tests for d in ["entier", "3s"]}
                                 | set(train["wav_path"])), model)
mu, _ = construire_reference(emb, train, moyenne=False)

print("Calcul des empreintes d'enrôlement...")
enrol = {(s, d, mode): (empreinte_signal(model, y[:d * SR]) if mode == "bloc"
                        else empreinte_morceaux(model, y[:d * SR]))
         for s, y in audio_enrol.items() for d in DUREES_ENROLEMENT for mode in ["bloc", "morceaux"]}

# 5. Évaluation : durée d'enrôlement × façon de calculer × réglage × durée de test
lignes = []
for d in DUREES_ENROLEMENT:
    for mode in ["bloc", "morceaux"]:
        for reglage in ["brut", "moyenne soustraite"]:
            m = mu if reglage != "brut" else None
            for duree_test in ["3s", "entier"]:
                T = preparer(np.stack([emb[version(f, duree_test)] for f, _ in tests]), m)
                rows = []
                for s in audio_enrol:
                    ref = preparer(enrol[(s, d, mode)], m)[0]
                    for score, st in zip(T @ ref, spk_test):
                        rows.append((score, int(st == s), frozenset((s, st)) in proches))
                t = pd.DataFrame(rows, columns=["score", "label", "proche"])
                for cas, sel in [("tout", t), ("difficile", t[(t.label == 1) | t.proche])]:
                    r = evaluate(sel["score"], sel["label"])
                    enregistrer(f"ECAPA enrôlement {d} s ({mode}) | {reglage} | test {duree_test} | {cas}",
                                "val_enrolement", r)
                    lignes.append({"enrôlement": f"{d:>2} s", "calcul": mode, "réglage": reglage,
                                   "test": duree_test, "négatives": cas, "EER": r["eer"],
                                   "minDCF": r["min_dcf"], "FRR@FA1%": r["frr_far1"],
                                   "seuil FA1%": r["seuil_far1"],
                                   "positifs": int(sel.label.sum()), "négatifs": int((sel.label == 0).sum())})

res = pd.DataFrame(lignes)
res.to_csv("results/enrolement_v2.csv", index=False)
n = res[(res["négatives"] == "difficile")].iloc[0]
print(f"Négatives difficiles : {n['positifs']} essais de la vraie personne, {n['négatifs']} essais d'imposteurs proches")

difficile_3s = res[(res["négatives"] == "difficile") & (res["test"] == "3s")]
for mesure in ["EER", "minDCF", "FRR@FA1%", "seuil FA1%"]:
    print(f"\n=== {mesure}, test 3 s, voix proches" + (" (plus bas = mieux)" if mesure != "seuil FA1%" else "") + " ===")
    print(difficile_3s.pivot_table(index="enrôlement", columns=["réglage", "calcul"], values=mesure)
          .round(3).to_string())