# src/bruit_v2.py
# Jour 5, étape 1 : robustesse au bruit, aux conditions de la démo.
#  - Enrôlement : 3 s propres (premier clip de la paire). Test : 3 s avec bruit ajouté (second clip).
#  - Deux bruits : "brouhaha" (plusieurs voix qui parlent en même temps, tirées des locuteurs de TRAIN)
#                  et "ambiance" (bruit de fond continu : ventilation, salle).
#  - Niveaux : 20 dB (léger), 10 dB (salle animée), 5 dB (fort). Plus le chiffre est bas, plus c'est bruyant.
#  - Configuration retenue : ECAPA + moyenne soustraite, seuil de la démo mesuré sur la validation.
# À lancer depuis la racine : python src/bruit_v2.py   (environ 15 à 20 minutes)
import numpy as np
import pandas as pd
import librosa
import torch
from evaluate import evaluate, enregistrer
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, preparer, construire_reference, seuil_validation

SR, GRAINE, LOT = 16000, 42, 32
NIVEAUX_DB = [20, 10, 5]
SEUIL = seuil_validation()
rng = np.random.default_rng(GRAINE)


def bruit_rose(n):
    """Bruit de fond continu, plus fort dans les graves (comme une ventilation ou une salle)."""
    spectre = np.fft.rfft(rng.normal(size=n))
    f = np.arange(len(spectre))
    spectre[1:] /= np.sqrt(f[1:])
    b = np.fft.irfft(spectre, n)
    return b / (np.abs(b).max() + 1e-9)


def brouhaha(n, voix):
    """Plusieurs voix de TRAIN superposées, à des moments différents (comme des gens qui discutent autour)."""
    b = np.zeros(n)
    for v in rng.choice(len(voix), 4, replace=False):
        y = voix[v]
        debut = rng.integers(0, max(1, len(y) - n))
        morceau = y[debut:debut + n]
        b[:len(morceau)] += morceau / (np.std(morceau) + 1e-9)
    return b


def melanger(y, b, snr_db):
    """Ajoute le bruit b au signal y au rapport signal/bruit voulu (en dB)."""
    p_signal, p_bruit = np.mean(y ** 2), np.mean(b ** 2) + 1e-12
    return y + b * np.sqrt(p_signal / (p_bruit * 10 ** (snr_db / 10)))


def empreintes_signaux(model, signaux):
    """Empreintes ECAPA de plusieurs signaux, calculées par lots (plus rapide sur CPU)."""
    out = []
    for i in range(0, len(signaux), LOT):
        lot = signaux[i:i + LOT]
        n = max(len(y) for y in lot)
        x = np.zeros((len(lot), n), dtype=np.float32)
        for j, y in enumerate(lot):
            x[j, :len(y)] = y
        longueurs = torch.tensor([len(y) / n for y in lot], dtype=torch.float32)
        with torch.no_grad():
            e = model.encode_batch(torch.tensor(x), longueurs)
        out.append(e.reshape(len(lot), -1).numpy())
    return np.vstack(out)


# Paires 3 s du banc v2 : positives et voix proches (le cas qui compte), plus négatives au hasard
bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
paires = bench[bench["duree"] == "3s"].reset_index(drop=True)
model = charger_modele()
train = clips_train()
emb = calculer_empreintes(sorted(set(paires[["file1", "file2"]].values.ravel()) | set(train["wav_path"])), model)
mu, _ = construire_reference(emb, train, moyenne=False)
A = preparer(np.stack([emb[f] for f in paires["file1"]]), mu)          # enrôlements propres

voix_train = [librosa.load(f, sr=SR)[0] for f in train["wav_path"].sample(60, random_state=GRAINE)]
tests = sorted(set(paires["file2"]))
propres = {f: librosa.load(f, sr=SR)[0] for f in tests}
print(f"{len(paires)} paires, {len(tests)} extraits de test, seuil de la démo = {SEUIL:.3f}")

conditions = [("propre", None)] + [(b, d) for b in ["ambiance", "brouhaha"] for d in NIVEAUX_DB]
lignes = []
for nom_bruit, db in conditions:
    if db is None:
        B_emb = {f: emb[f] for f in tests}
    else:
        print(f"Bruit {nom_bruit} à {db} dB ...")
        bruites = [melanger(propres[f], bruit_rose(len(propres[f])) if nom_bruit == "ambiance"
                            else brouhaha(len(propres[f]), voix_train), db) for f in tests]
        B_emb = dict(zip(tests, empreintes_signaux(model, bruites)))
    B = preparer(np.stack([B_emb[f] for f in paires["file2"]]), mu)
    paires["score"] = (A * B).sum(axis=1)

    for cas, negs in [("difficile", ["negative_difficile"]), ("tout", ["negative_hasard", "negative_difficile"])]:
        sel = paires[(paires["type"] == "positive") | paires["type"].isin(negs)]
        r = evaluate(sel["score"], sel["label"])
        vrais, faux = sel[sel["label"] == 1], sel[sel["label"] == 0]
        condition = "propre" if db is None else f"{nom_bruit} {db} dB"
        enregistrer(f"ECAPA moyenne soustraite | 3 s / 3 s | {condition} | {cas}", "val_bruit", r)
        lignes.append({"condition": condition, "négatives": cas, "EER": r["eer"], "minDCF": r["min_dcf"],
                       "vraie personne refusée (seuil démo)": (vrais["score"] < SEUIL).mean(),
                       "imposteur accepté (seuil démo)": (faux["score"] >= SEUIL).mean(),
                       "score moyen vraie personne": vrais["score"].mean()})

res = pd.DataFrame(lignes)
res.to_csv("results/bruit_v2.csv", index=False)
print("\n=== Voix proches, enrôlement 3 s propre, test 3 s bruité ===")
print(res[res["négatives"] == "difficile"].drop(columns="négatives").round(3).to_string(index=False))