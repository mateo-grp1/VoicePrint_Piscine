# src/metric_learning.py
# Jour 6 : metric learning léger au-dessus d'ECAPA (ECAPA reste gelé).
#
# Idée : une petite couche apprend à rapprocher les empreintes d'une même personne et à éloigner celles
# de personnes différentes, dans les conditions difficiles de la démo :
#   - extraits de 3 s pris dans des PHRASES DIFFÉRENTES (donc des sons différents, comme un changement de langue),
#   - avec ou sans bruit (ambiance, brouhaha).
# La couche part de "ne rien changer" (sortie = entrée) : à l'epoch 0, on retrouve exactement ECAPA.
# L'entraînement ne garde que ce qui améliore la VALIDATION ; si rien n'aide, on garde ECAPA seul.
#
# Étapes : 1. empreintes d'entraînement augmentées (mises en cache, ~10-15 min la 1re fois)
#          2. empreintes de validation propres et bruitées (cache)
#          3. entraînement (triplet loss, triplets "difficiles" choisis dans chaque lot), quelques minutes
#          4. comparaison ECAPA seul / ECAPA + couche, courbes, sauvegarde de la couche
# À lancer depuis la racine : python src/metric_learning.py
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluate import evaluate, enregistrer
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, preparer, construire_reference
from augmentation import bruit_rose, brouhaha, melanger, empreintes_signaux

SR, DUREE, GRAINE = 16000, 3, 42
CLIPS_PAR_LOC = 20                 # clips d'entraînement par locuteur (plafond)
VARIANTES = [("propre", None), ("propre", None), ("ambiance", 20), ("ambiance", 10),
             ("brouhaha", 20), ("brouhaha", 10)]   # 6 versions de chaque clip, chacune avec un extrait de 3 s différent
EPOCHS, P, K = 40, 16, 6           # un lot = 16 personnes x 6 empreintes
MARGE, LR, WD = 0.2, 1e-3, 1e-3
CACHE_TRAIN = Path("data/embeddings/ml_train.npz")
CACHE_VAL = Path("data/embeddings/ml_val_bruit.npz")
rng = np.random.default_rng(GRAINE)
torch.manual_seed(GRAINE)


def charger(f):
    y, _ = librosa.load(f, sr=SR)
    return librosa.effects.trim(y, top_db=30)[0]


def extrait(y, n=DUREE * SR):
    debut = rng.integers(0, max(1, len(y) - n + 1))
    return y[debut:debut + n]


def bruiter(y, type_bruit, db, voix):
    if type_bruit == "propre":
        return y
    b = bruit_rose(len(y), rng) if type_bruit == "ambiance" else brouhaha(len(y), voix, rng)
    return melanger(y, b, db)


model = charger_modele()
meta = pd.read_csv("data/metadata.csv")
train_meta = meta[meta["split"] == "train"]
train_meta = train_meta.sample(frac=1, random_state=GRAINE).groupby("speaker_id").head(CLIPS_PAR_LOC)
train_meta = train_meta[train_meta.groupby("speaker_id")["wav_path"].transform("size") >= 2]

# Moyenne à soustraire (même référence que la démo)
ref_train = clips_train()
emb_cache = calculer_empreintes(sorted(set(ref_train["wav_path"])), model)
mu, _ = construire_reference(emb_cache, ref_train, moyenne=False)

# ---------- 1. Empreintes d'entraînement augmentées ----------
if CACHE_TRAIN.exists():
    d = np.load(CACHE_TRAIN, allow_pickle=True)
    X_tr, spk_tr = d["X"], d["spk"]
else:
    print(f"Empreintes d'entraînement : {len(train_meta)} clips x {len(VARIANTES)} versions (une seule fois)...")
    voix = [charger(f) for f in train_meta["wav_path"].sample(min(60, len(train_meta)), random_state=GRAINE)]
    signaux, spk_tr = [], []
    for n, (f, s) in enumerate(zip(train_meta["wav_path"], train_meta["speaker_id"]), 1):
        y = charger(f)
        for type_bruit, db in VARIANTES:
            signaux.append(bruiter(extrait(y), type_bruit, db, voix))
            spk_tr.append(s)
        if n % 100 == 0:
            print(f"  {n}/{len(train_meta)} clips préparés")
    print("  calcul des empreintes...")
    X_tr, spk_tr = empreintes_signaux(model, signaux), np.array(spk_tr)
    CACHE_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_TRAIN, X=X_tr, spk=spk_tr)
X_tr = preparer(X_tr, mu)
print(f"Entraînement : {len(np.unique(spk_tr))} locuteurs, {len(X_tr)} empreintes")

# ---------- 2. Validation : paires 3 s du banc v2, test propre et bruité à 10 dB ----------
bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
paires = bench[bench["duree"] == "3s"].reset_index(drop=True)
emb_val = calculer_empreintes(sorted(set(paires[["file1", "file2"]].values.ravel())), model)
tests = sorted(set(paires["file2"]))
if CACHE_VAL.exists():
    d = np.load(CACHE_VAL, allow_pickle=True)
    val_bruit = {k: dict(zip(d["files"], d[k])) for k in ["ambiance", "brouhaha"]}
else:
    print("Empreintes de validation bruitées (une seule fois)...")
    voix_val = [charger(f) for f in ref_train["wav_path"].sample(60, random_state=GRAINE + 1)]
    val_bruit = {}
    for type_bruit in ["ambiance", "brouhaha"]:
        sig = [bruiter(librosa.load(f, sr=SR)[0], type_bruit, 10, voix_val) for f in tests]
        val_bruit[type_bruit] = dict(zip(tests, empreintes_signaux(model, sig)))
    np.savez(CACHE_VAL, files=np.array(tests),
             **{k: np.stack([v[f] for f in tests]) for k, v in val_bruit.items()})

A_val = preparer(np.stack([emb_val[f] for f in paires["file1"]]), mu)
B_val = {"propre": preparer(np.stack([emb_val[f] for f in paires["file2"]]), mu),
         **{k: preparer(np.stack([v[f] for f in paires["file2"]]), mu) for k, v in val_bruit.items()}}
difficile = paires["type"].isin(["positive", "negative_difficile"]).values
labels = paires["label"].values


# ---------- 3. La couche et son entraînement ----------
class Couche(nn.Module):
    """Sortie = entrée + correction apprise. La correction part de zéro : au départ, rien ne change."""
    def __init__(self, d=192):
        super().__init__()
        self.lin = nn.Linear(d, d)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return F.normalize(x + self.lin(x), dim=-1)


def appliquer(couche, X):
    with torch.no_grad():
        return couche(torch.tensor(X, dtype=torch.float32)).numpy()


def mesurer(couche, condition, sel=difficile):
    A, B = appliquer(couche, A_val), appliquer(couche, B_val[condition])
    s = (A * B).sum(axis=1)
    return evaluate(s[sel], labels[sel])


def perte_triplets(E, spk):
    """Triplet loss "batch-hard" : pour chaque ancre, le positif le plus éloigné et le négatif le plus proche du lot."""
    dist = 1 - E @ E.T
    meme = spk[:, None] == spk[None, :]
    pos = (dist * meme.float()).max(dim=1).values
    neg = (dist + 10 * meme.float()).min(dim=1).values
    return F.relu(pos - neg + MARGE).mean()


par_loc = {s: np.where(spk_tr == s)[0] for s in np.unique(spk_tr)}
locs = np.array(list(par_loc))
X_t = torch.tensor(X_tr, dtype=torch.float32)
spk_code = torch.tensor(pd.factorize(spk_tr)[0])


def lots():
    """Lots de P personnes x K empreintes chacune (ce qui garantit des positifs et des négatifs dans chaque lot)."""
    ordre = rng.permutation(locs)
    for i in range(0, len(ordre) - P + 1, P):
        idx = np.concatenate([rng.choice(par_loc[s], K, replace=len(par_loc[s]) < K) for s in ordre[i:i + P]])
        yield torch.tensor(idx)


couche = Couche()
opt = torch.optim.Adam(couche.parameters(), lr=LR, weight_decay=WD)
r0p, r0b = mesurer(couche, "propre"), mesurer(couche, "brouhaha")
critere0 = (r0p["min_dcf"] + r0b["min_dcf"]) / 2
meilleur = {"epoch": 0, "critere": critere0, "poids": copy.deepcopy(couche.state_dict())}
hist = {"loss": [np.nan], "mindcf_propre": [r0p["min_dcf"]], "mindcf_brouhaha": [r0b["min_dcf"]]}
print(f"\nEpoch  0 (ECAPA seul) | minDCF voix proches : propre {r0p['min_dcf']:.3f}, brouhaha 10 dB {r0b['min_dcf']:.3f}")

for epoch in range(1, EPOCHS + 1):
    couche.train()
    pertes = []
    for idx in lots():
        opt.zero_grad()
        perte = perte_triplets(couche(X_t[idx]), spk_code[idx])
        perte.backward()
        opt.step()
        pertes.append(perte.item())
    couche.eval()
    rp, rb = mesurer(couche, "propre"), mesurer(couche, "brouhaha")
    critere = (rp["min_dcf"] + rb["min_dcf"]) / 2
    hist["loss"].append(np.mean(pertes))
    hist["mindcf_propre"].append(rp["min_dcf"])
    hist["mindcf_brouhaha"].append(rb["min_dcf"])
    if critere < meilleur["critere"]:
        meilleur = {"epoch": epoch, "critere": critere, "poids": copy.deepcopy(couche.state_dict())}
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | loss {np.mean(pertes):.3f} | minDCF propre {rp['min_dcf']:.3f}, "
              f"brouhaha {rb['min_dcf']:.3f}")

# ---------- 4. Bilan ----------
Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["loss"], color="#2a78d6")
ax[0].set_title("Loss d'entraînement (triplets)", loc="left")
ax[0].set_xlabel("epoch")
ax[1].plot(hist["mindcf_propre"], color="#2a78d6", label="test propre")
ax[1].plot(hist["mindcf_brouhaha"], color="#eb6834", label="test brouhaha 10 dB")
ax[1].axvline(meilleur["epoch"], color="#52514e", linestyle="--", linewidth=1, label="epoch gardée")
ax[1].set_title("minDCF sur la validation (voix proches, 3 s)", loc="left")
ax[1].set_xlabel("epoch (0 = ECAPA seul)")
ax[1].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.savefig("results/metric_learning_courbes.png", dpi=150, bbox_inches="tight")
plt.close()

ecapa_seul = Couche()
couche.load_state_dict(meilleur["poids"])
lignes = []
for nom, c in [("ECAPA + moyenne soustraite", ecapa_seul), ("+ couche metric learning", couche)]:
    for condition in ["propre", "ambiance", "brouhaha"]:
        r = mesurer(c, condition)
        enregistrer(f"{nom} | 3 s / 3 s | {condition}{'' if condition == 'propre' else ' 10 dB'} | difficile",
                    "val_metric", r)
        lignes.append({"méthode": nom, "test": condition if condition == "propre" else f"{condition} 10 dB",
                       "EER": r["eer"], "minDCF": r["min_dcf"], "FRR@FA1%": r["frr_far1"],
                       "seuil FA1%": r["seuil_far1"]})

print(f"\nEpoch gardée : {meilleur['epoch']} sur {EPOCHS}" + ("  (0 = la couche n'a rien amélioré, on garde ECAPA seul)"
                                                          if meilleur["epoch"] == 0 else ""))
t = pd.DataFrame(lignes)
for mesure in ["minDCF", "FRR@FA1%", "EER", "seuil FA1%"]:
    print(f"\n=== {mesure} — validation, voix proches, enrôlement 3 s propre / test 3 s"
          + ("" if mesure == "seuil FA1%" else " (plus bas = mieux)") + " ===")
    print(t.pivot_table(index="test", columns="méthode", values=mesure)
          [["ECAPA + moyenne soustraite", "+ couche metric learning"]].round(3).to_string())

Path("models").mkdir(exist_ok=True)
torch.save({"poids": meilleur["poids"], "epoch": meilleur["epoch"]}, "models/couche_metric.pt")
print("\nCouche sauvegardée : models/couche_metric.pt | courbes : results/metric_learning_courbes.png")