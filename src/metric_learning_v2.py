# src/metric_learning_v2.py
# Jour 6-7 : metric learning v2 au-dessus d'ECAPA gelé, entraîné sur Common Voice 27.0 (945 locuteurs).
#
# Pourquoi une v2 : la v1 (85 locuteurs, triplet loss, couche 192x192 libre) a sur-appris.
# La v2 corrige les 4 causes identifiées :
#   1. Plus de locuteurs : 945 personnes de cv27 (aucune de la validation ni du test cv26, vérifié).
#   2. Une loss plus adaptée : AAM-softmax (celle avec laquelle ECAPA a été entraîné), au lieu des triplets.
#   3. Une couche plus contrainte, ramenée vers "ne rien changer" par une pénalité :
#        - "diagonale"      : un poids par dimension (192 paramètres) = donner plus ou moins d'importance
#                             à chaque caractéristique d'ECAPA ;
#        - "residuelle_r32" : sortie = entrée + petite correction de rang 32 (environ 12 000 paramètres).
#   4. Un critère de choix d'epoch stable : minDCF moyen sur TOUTES les paires 3 s de validation
#      (propres, ambiance 10 dB, brouhaha 10 dB), lissé sur 3 epochs.
# En plus : 10 % des locuteurs cv27 sont tenus à l'écart de l'entraînement pour SURVEILLER l'overfit
# (si leur EER remonte pendant que la loss baisse, la couche apprend par cœur).
#
# Inchangé : ECAPA gelé, moyenne soustraite (même référence que la démo), couche qui part de
# "ne rien changer", même banc de validation que la v1, règle "gardée seulement si elle améliore la validation".
#
# Prérequis : data/cv27/emb_cv27_train.npz (préparé sur Colab), caches de la v1 (data/embeddings/).
# À lancer depuis la racine : python src/metric_learning_v2.py
import copy
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentation import brouhaha, bruit_rose, empreintes_signaux, melanger
from evaluate import enregistrer, evaluate
from normalisation import clips_train, construire_reference, preparer

GRAINE, SR = 42, 16000
COUCHES = ["diagonale", "residuelle_r32"]
EPOCHS, LOT, LR = 30, 512, 1e-3
AAM_MARGE, AAM_ECHELLE = 0.2, 30.0      # mêmes valeurs que l'entraînement d'ECAPA
PENALITE = 1e-2                          # rappel vers "ne rien changer"
PART_TENUS_A_L_ECART = 0.10
LISSAGE = 3

DOSSIER_CV27 = Path("data/cv27")
CACHE_ECAPA = Path("data/embeddings/ecapa.npz")
CACHE_VAL = Path("data/embeddings/ml_val_bruit.npz")   # même validation bruitée que la v1
rng = np.random.default_rng(GRAINE)
torch.manual_seed(GRAINE)

_modele = None


def modele():
    """ECAPA n'est chargé que s'il manque des empreintes dans les caches."""
    global _modele
    if _modele is None:
        from ecapa_eval import charger_modele
        _modele = charger_modele()
    return _modele


def empreintes(chemins):
    """Empreintes ECAPA des fichiers : depuis le cache, ou calculées si absentes."""
    chemins = sorted(set(chemins))
    if CACHE_ECAPA.exists():
        d = np.load(CACHE_ECAPA, allow_pickle=True)
        cache = dict(zip(d["paths"], d["emb"]))
        if all(c in cache for c in chemins):
            return cache
    from ecapa_eval import calculer_empreintes
    return calculer_empreintes(chemins, modele())


# ---------- 1. Moyenne à soustraire : la même que la démo (train cv26) ----------
ref_train = clips_train()
mu, _ = construire_reference(empreintes(ref_train["wav_path"]), ref_train, moyenne=False)

# ---------- 2. Données d'entraînement cv27 ----------
# allow_pickle=True : les noms de clips ont été enregistrés sur Colab comme objets Python.
# Sans risque ici, car c'est notre propre fichier (jamais un fichier venu d'ailleurs).
d = np.load(DOSSIER_CV27 / "emb_cv27_train.npz", allow_pickle=True)
X_all, spk_all, ver_all, clip_all = d["emb"], d["locuteur"], d["version"], d["clip"].astype(str)
fichier_retrait = DOSSIER_CV27 / "locuteurs_a_retirer.csv"
if fichier_retrait.exists():
    retires = pd.read_csv(fichier_retrait)["locuteur"].str[1:].astype(int).to_numpy()
    garder = ~np.isin(spk_all, retires)
    X_all, spk_all, ver_all, clip_all = X_all[garder], spk_all[garder], ver_all[garder], clip_all[garder]
    print(f"Locuteurs retirés (anti-fuite) : {len(retires)}")
X_all = preparer(X_all, mu)

locs = np.unique(spk_all)
ecart = rng.choice(locs, size=int(len(locs) * PART_TENUS_A_L_ECART), replace=False)
m_ecart = np.isin(spk_all, ecart)
X_tr, spk_tr = X_all[~m_ecart], spk_all[~m_ecart]
X_ec, spk_ec, ver_ec, clip_ec = X_all[m_ecart], spk_all[m_ecart], ver_all[m_ecart], clip_all[m_ecart]
classes, y_tr = np.unique(spk_tr, return_inverse=True)
print(f"Entraînement : {len(classes)} locuteurs, {len(X_tr)} empreintes | "
      f"tenus à l'écart (surveillance) : {len(ecart)} locuteurs, {len(X_ec)} empreintes")

# Paires des locuteurs tenus à l'écart : enrôlement propre (version 0) contre test propre ou bruité 10 dB,
# toujours deux clips (donc deux phrases) différents
idx_ec = np.arange(len(X_ec))
enrol = idx_ec[ver_ec == 0]
paires_ec = {"propre": 1, "ambiance": 3, "brouhaha": 5}
paires_ec_idx = {}
for nom, v in paires_ec.items():
    tests = idx_ec[ver_ec == v]
    a = rng.choice(enrol, 3000)
    pos_b = np.array([rng.choice(tests[(spk_ec[tests] == spk_ec[i]) & (clip_ec[tests] != clip_ec[i])]) for i in a[:1500]])
    neg_b = np.array([rng.choice(tests[spk_ec[tests] != spk_ec[i]]) for i in a[1500:]])
    paires_ec_idx[nom] = (a, np.concatenate([pos_b, neg_b]), np.r_[np.ones(1500), np.zeros(1500)])

# ---------- 3. Validation cv26 : même banc que la v1 (3 s, propre / ambiance 10 dB / brouhaha 10 dB) ----------
paires = pd.read_csv("data/pairs/pairs_val_v2.csv")
paires = paires[paires["duree"] == "3s"].reset_index(drop=True)
emb_val = empreintes(paires[["file1", "file2"]].values.ravel())
tests_val = sorted(set(paires["file2"]))
if CACHE_VAL.exists():
    dv = np.load(CACHE_VAL, allow_pickle=True)
    val_bruit = {k: dict(zip(dv["files"], dv[k])) for k in ["ambiance", "brouhaha"]}
else:   # même recette que la v1
    print("Empreintes de validation bruitées (une seule fois)...")
    voix_val = [librosa.effects.trim(librosa.load(f, sr=SR)[0], top_db=30)[0]
                for f in ref_train["wav_path"].sample(60, random_state=GRAINE + 1)]
    val_bruit = {}
    for type_bruit in ["ambiance", "brouhaha"]:
        sig = []
        for f in tests_val:
            y = librosa.load(f, sr=SR)[0]
            b = bruit_rose(len(y), rng) if type_bruit == "ambiance" else brouhaha(len(y), voix_val, rng)
            sig.append(melanger(y, b, 10))
        val_bruit[type_bruit] = dict(zip(tests_val, empreintes_signaux(modele(), sig)))
    np.savez(CACHE_VAL, files=np.array(tests_val), **{k: np.stack([v[f] for f in tests_val]) for k, v in val_bruit.items()})

A_val = preparer(np.stack([emb_val[f] for f in paires["file1"]]), mu)
B_val = {"propre": preparer(np.stack([emb_val[f] for f in paires["file2"]]), mu),
         **{k: preparer(np.stack([v[f] for f in paires["file2"]]), mu) for k, v in val_bruit.items()}}
labels = paires["label"].values
difficile = paires["type"].isin(["positive", "negative_difficile"]).values
CONDITIONS = ["propre", "ambiance", "brouhaha"]


# ---------- 4. Les couches et la loss ----------
class Diagonale(nn.Module):
    """Un poids par dimension d'ECAPA (192 paramètres). Au départ : tous à 1 = rien ne change."""
    def __init__(self, d=192):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.normalize(x * self.w, dim=-1)

    def penalite(self):
        return ((self.w - 1) ** 2).sum()


class ResiduelleRang(nn.Module):
    """Sortie = entrée + correction de rang r (x -> 192 -> r -> 192). Au départ : correction nulle."""
    def __init__(self, d=192, r=32):
        super().__init__()
        self.bas = nn.Linear(d, r, bias=False)
        self.haut = nn.Linear(r, d)
        nn.init.zeros_(self.haut.weight)
        nn.init.zeros_(self.haut.bias)

    def forward(self, x):
        return F.normalize(x + self.haut(self.bas(x)), dim=-1)

    def penalite(self):
        return (self.haut.weight ** 2).sum() + (self.haut.bias ** 2).sum()


def creer(nom):
    return Diagonale() if nom == "diagonale" else ResiduelleRang(r=32)


class AAMSoftmax(nn.Module):
    """Chaque personne d'entraînement a un "centre". La loss pousse chaque empreinte vers SON centre
    avec une marge d'angle : il ne suffit pas d'être le plus proche, il faut l'être nettement."""
    def __init__(self, centres, marge=AAM_MARGE, echelle=AAM_ECHELLE):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(centres, dtype=torch.float32))
        self.cos_m, self.sin_m, self.s = np.cos(marge), np.sin(marge), echelle
        self.seuil, self.mm = np.cos(np.pi - marge), np.sin(np.pi - marge) * marge

    def forward(self, E, y):
        cos = F.linear(E, F.normalize(self.W, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1 - cos ** 2)
        phi = cos * self.cos_m - sin * self.sin_m                     # cos(angle + marge)
        phi = torch.where(cos > self.seuil, phi, cos - self.mm)
        un = F.one_hot(y, cos.shape[1]).bool()
        return F.cross_entropy(self.s * torch.where(un, phi, cos), y)


def appliquer(couche, X):
    with torch.no_grad():
        return couche(torch.tensor(X, dtype=torch.float32)).numpy()


def mesurer_val(couche, condition, sel=None):
    A, B = appliquer(couche, A_val), appliquer(couche, B_val[condition])
    s = (A * B).sum(axis=1)
    sel = np.ones(len(s), bool) if sel is None else sel
    return evaluate(s[sel], labels[sel])


def mesurer_ecart(couche, condition):
    a, b, lab = paires_ec_idx[condition]
    E = appliquer(couche, X_ec)
    return evaluate((E[a] * E[b]).sum(axis=1), lab)["eer"]


def critere(couche):
    """Stable : minDCF moyen sur toutes les paires 3 s (hasard + voix proches), dans les 3 conditions."""
    return float(np.mean([mesurer_val(couche, c)["min_dcf"] for c in CONDITIONS]))


# Centres de départ = empreinte moyenne de chaque personne (la loss démarre déjà dans le bon sens)
centres = np.stack([X_tr[y_tr == k].mean(axis=0) for k in range(len(classes))])
X_t, y_t = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr)


def entrainer(nom):
    torch.manual_seed(GRAINE)
    couche, tete = creer(nom), AAMSoftmax(centres)
    opt = torch.optim.Adam([{"params": couche.parameters(), "lr": LR},
                            {"params": tete.parameters(), "lr": LR * 10}])
    hist = {"loss": [np.nan], "critere": [critere(couche)],
            **{f"val_{c}": [mesurer_val(couche, c, difficile)["min_dcf"]] for c in CONDITIONS},
            **{f"ecart_{c}": [mesurer_ecart(couche, c)] for c in CONDITIONS}}
    etats = [copy.deepcopy(couche.state_dict())]
    print(f"\n=== Couche {nom} ({sum(p.numel() for p in couche.parameters())} paramètres) ===")
    print(f"Epoch  0 (ECAPA seul) | critère {hist['critere'][0]:.3f} | "
          f"EER tenus à l'écart (propre) {hist['ecart_propre'][0]:.3f}")
    for epoch in range(1, EPOCHS + 1):
        couche.train()
        pertes = []
        for i in torch.randperm(len(X_t)).split(LOT):
            opt.zero_grad()
            perte = tete(couche(X_t[i]), y_t[i]) + PENALITE * couche.penalite()
            perte.backward()
            opt.step()
            pertes.append(perte.item())
        couche.eval()
        hist["loss"].append(np.mean(pertes))
        hist["critere"].append(critere(couche))
        for c in CONDITIONS:
            hist[f"val_{c}"].append(mesurer_val(couche, c, difficile)["min_dcf"])
            hist[f"ecart_{c}"].append(mesurer_ecart(couche, c))
        etats.append(copy.deepcopy(couche.state_dict()))
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | loss {hist['loss'][-1]:.3f} | critère {hist['critere'][-1]:.3f} | "
                  f"EER tenus à l'écart : propre {hist['ecart_propre'][-1]:.3f}, brouhaha {hist['ecart_brouhaha'][-1]:.3f}")

    # Choix de l'epoch sur le critère LISSÉ (moyenne glissante) : on évite de garder un pic de chance
    c = pd.Series(hist["critere"]).rolling(LISSAGE, center=True, min_periods=1).mean().to_numpy()
    meilleure = int(np.argmin(c))
    couche.load_state_dict(etats[meilleure])
    return couche, hist, meilleure


resultats, courbes = {}, {}
for nom in COUCHES:
    couche, hist, ep = entrainer(nom)
    resultats[nom], courbes[nom] = (couche, ep), hist
    print(f"Epoch gardée : {ep}" + ("  (0 = la couche n'améliore rien : on garde ECAPA seul)" if ep == 0 else ""))

# ---------- 5. Courbes ----------
Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(len(COUCHES), 3, figsize=(15, 4 * len(COUCHES)), squeeze=False)
couleurs = {"propre": "#2a78d6", "ambiance": "#1baf7a", "brouhaha": "#eb6834"}
for r, nom in enumerate(COUCHES):
    h, ep = courbes[nom], resultats[nom][1]
    ax[r, 0].plot(h["loss"], color="#2a78d6")
    ax[r, 0].set_title(f"{nom} : loss d'entraînement (AAM-softmax)", loc="left")
    for c in CONDITIONS:
        ax[r, 1].plot(h[f"ecart_{c}"], color=couleurs[c], label=c)
        ax[r, 2].plot(h[f"val_{c}"], color=couleurs[c], label=c)
    ax[r, 1].set_title("EER des locuteurs cv27 tenus à l'écart (surveillance)", loc="left")
    ax[r, 2].set_title("minDCF validation cv26 (voix proches, 3 s)", loc="left")
    for a in ax[r]:
        a.axvline(ep, color="#52514e", linestyle="--", linewidth=1)
        a.set_xlabel("epoch (0 = ECAPA seul)")
        a.spines[["top", "right"]].set_visible(False)
    ax[r, 1].legend(frameon=False)
plt.tight_layout()
plt.savefig("results/metric_learning_v2_courbes.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 6. Bilan : mêmes paires, même fonction d'évaluation ----------
lignes = []
methodes = [("ECAPA + moyenne soustraite", Diagonale())] + [(f"+ couche v2 {n}", resultats[n][0]) for n in COUCHES]
for nom, c in methodes:
    for condition in CONDITIONS:
        r = mesurer_val(c, condition, difficile)
        test = condition if condition == "propre" else f"{condition} 10 dB"
        enregistrer(f"{nom} | 3 s / 3 s | {test} | difficile", "val_metric_v2", r,
                    notes="ECAPA gelé ; couche entraînée sur 945 locuteurs cv27, AAM-softmax")
        lignes.append({"méthode": nom, "test": test, "EER": r["eer"], "minDCF": r["min_dcf"],
                       "FRR@FA1%": r["frr_far1"], "seuil FA1%": r["seuil_far1"]})
t = pd.DataFrame(lignes)
ordre = [m for m, _ in methodes]
for mesure in ["minDCF", "FRR@FA1%", "EER", "seuil FA1%"]:
    print(f"\n=== {mesure} — validation, voix proches, enrôlement 3 s propre / test 3 s"
          + ("" if mesure == "seuil FA1%" else " (plus bas = mieux)") + " ===")
    print(t.pivot_table(index="test", columns="méthode", values=mesure)[ordre].round(3).to_string())

base = t[t["méthode"] == ordre[0]]
print("\n=== Verdict (règle : gardée seulement si elle améliore la validation) ===")
for nom in COUCHES:
    v = t[t["méthode"] == f"+ couche v2 {nom}"]
    gain_dcf = base["minDCF"].mean() - v["minDCF"].mean()
    gain_frr = base["FRR@FA1%"].mean() - v["FRR@FA1%"].mean()
    pire_propre = v[v["test"] == "propre"]["minDCF"].iloc[0] - base[base["test"] == "propre"]["minDCF"].iloc[0]
    ok = gain_dcf > 0 and gain_frr > 0 and pire_propre <= 0.01
    print(f"{nom:<16} minDCF moyen {gain_dcf:+.3f} | FRR@FA1% moyen {gain_frr:+.1%} | "
          f"écart en propre {pire_propre:+.3f} -> {'AMÉLIORE ✅' if ok else 'pas d’amélioration nette ❌'}")
    Path("models").mkdir(exist_ok=True)
    torch.save({"type": nom, "poids": resultats[nom][0].state_dict(), "epoch": resultats[nom][1]},
               f"models/couche_metric_v2_{nom}.pt")
print("\nCourbes : results/metric_learning_v2_courbes.png | couches : models/couche_metric_v2_*.pt")