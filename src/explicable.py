# src/explicable.py
# Jour 4, étape 2 : qu'est-ce qui différencie vraiment deux voix ?
#  1. Mesure les caractéristiques vocales nommées (hauteur, formants, timbre...) sur chaque clip.
#  2. Ratio de Fisher de chacune : variation ENTRE personnes / variation CHEZ une même personne
#     (calculé sur les locuteurs d'ENTRAÎNEMENT).
#  3. Test du plus proche voisin sur la validation : avec ECAPA, puis avec les caractéristiques nommées,
#     le clip le plus ressemblant vient-il de la même personne ?
#  4. Carte t-SNE des empreintes ECAPA (validation + mini-corpus).
# À lancer depuis la racine : python src/explicable.py
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from ecapa_eval import charger_modele, calculer_empreintes
from normalisation import clips_train, preparer, construire_reference
from caracteristiques import extraire, COLONNES, DESCRIPTION

SR = 16000
CACHE = Path("data/features/caracteristiques.csv")
# Couleurs (palette catégorielle validée : bleu, orange, aqua ; textes et surface)
BLEU, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, TEXTE, TEXTE_2, GRILLE, GRIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRILLE,
                     "axes.labelcolor": TEXTE_2, "xtick.color": TEXTE_2, "ytick.color": TEXTE,
                     "text.color": TEXTE, "font.size": 10})


def parole(f):
    y, _ = librosa.load(f, sr=SR)
    return librosa.effects.trim(y, top_db=30)[0]


# ---------- Données : clips de train (Fisher) et de validation (voisins, t-SNE) ----------
train = clips_train()[["wav_path", "speaker_id"]].rename(columns={"wav_path": "file", "speaker_id": "spk"})
bench = pd.read_csv("data/pairs/pairs_val_v2.csv")
entier = bench[bench["duree"] == "entier"]
val = pd.concat([entier[["file1", "spk1"]].set_axis(["file", "spk"], axis=1),
                 entier[["file2", "spk2"]].set_axis(["file", "spk"], axis=1)]).drop_duplicates("file")
train["groupe"], val["groupe"] = "train", "val"
clips = pd.concat([train, val], ignore_index=True)

# ---------- 1. Caractéristiques (mises en cache : le calcul prend quelques minutes) ----------
deja = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["file"] + COLONNES)
a_faire = [f for f in clips["file"] if f not in set(deja["file"])]
print(f"Caractéristiques : {len(deja)} clips en cache, {len(a_faire)} à mesurer")
nouvelles = []
for n, f in enumerate(a_faire, 1):
    nouvelles.append({"file": f, **extraire(parole(f))})
    if n % 100 == 0:
        print(f"  {n}/{len(a_faire)}")
if nouvelles:
    deja = pd.concat([deja, pd.DataFrame(nouvelles)], ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    deja.to_csv(CACHE, index=False)
clips = clips.merge(deja, on="file")

# ---------- 2. Ratio de Fisher (train uniquement) ----------
def fisher(valeurs, locuteurs):
    d = pd.DataFrame({"v": valeurs, "spk": locuteurs}).dropna()
    g = d.groupby("spk")["v"]
    g = d[d["spk"].isin(g.size()[g.size() >= 2].index)].groupby("spk")["v"]
    return g.mean().var() / g.var().mean()

tr = clips[clips["groupe"] == "train"]
fis = pd.DataFrame([{"caractéristique": DESCRIPTION[c][0], "colonne": c, "décrit": DESCRIPTION[c][2],
                     "fisher": fisher(tr[c], tr["spk"])} for c in COLONNES])

# Référence ECAPA : ratio de Fisher de chacune de ses 192 dimensions (moyenne soustraite)
model = charger_modele()
emb = calculer_empreintes(sorted(set(clips["file"])), model)
mu, _ = construire_reference(emb, train.rename(columns={"file": "wav_path", "spk": "speaker_id"}), moyenne=False)
E_tr = preparer(np.stack([emb[f] for f in tr["file"]]), mu)
f_ecapa = np.array([fisher(E_tr[:, d], tr["spk"].values) for d in range(E_tr.shape[1])])
print(f"\nECAPA, ratio de Fisher de ses 192 dimensions : moyenne {f_ecapa.mean():.2f}, meilleure {f_ecapa.max():.2f}")

fis = fis.sort_values("fisher", ascending=False)
Path("results").mkdir(exist_ok=True)
fis.to_csv("results/fisher_caracteristiques.csv", index=False)
print("\n=== Ratio de Fisher (plus haut = distingue mieux les personnes) ===")
print(fis[["caractéristique", "décrit", "fisher"]].round(2).to_string(index=False))

# Graphique : une barre par caractéristique, triées
f_plot = fis.sort_values("fisher")
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(f_plot["caractéristique"], f_plot["fisher"], color=BLEU, height=0.7)
if f_plot["fisher"].max() / max(f_plot["fisher"].min(), 1e-9) > 30:
    ax.set_xscale("log")                      # valeurs très étalées : échelle logarithmique
ax.axvline(f_ecapa.mean(), color=TEXTE_2, linestyle="--", linewidth=1)
ax.text(f_ecapa.mean(), -1.3, f" ECAPA : moyenne de ses 192 dimensions ({f_ecapa.mean():.1f})",
        color=TEXTE_2, fontsize=9, va="center")
ax.set_ylim(-1.8, len(f_plot) - 0.4)
for i, v in enumerate(f_plot["fisher"]):
    ax.text(v, i, f" {v:.1f}", va="center", fontsize=8, color=TEXTE_2)
ax.set_xlabel("Ratio de Fisher = variation entre personnes / variation chez une même personne")
ax.set_title("Ce qui distingue le plus les personnes", loc="left", fontweight="bold")
ax.grid(axis="x", color=GRILLE, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.savefig("results/fisher_caracteristiques.png", dpi=150, bbox_inches="tight")
plt.close()

# Écart habituel chez une même personne (sert à l'explication de la démo)
intra = {c: tr.groupby("spk")[c].std().mean() for c in COLONNES}
pd.Series(intra, name="ecart_intra").to_csv("results/ecart_intra_locuteur.csv")

# ---------- 3. Plus proche voisin sur la validation ----------
va = clips[clips["groupe"] == "val"].reset_index(drop=True)
spk = va["spk"].values

def taux_voisin(X, cosinus):
    if cosinus:
        S = X @ X.T
    else:
        S = -((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(S, -np.inf)
    return (spk[S.argmax(axis=1)] == spk).mean()

E_va = preparer(np.stack([emb[f] for f in va["file"]]), mu)
cols = [c for c in COLONNES if va[c].notna().all() and tr[c].notna().any()]
X_tr = tr[cols].fillna(tr[cols].median())
X_va = ((va[cols] - X_tr.mean()) / X_tr.std()).values     # standardisation apprise sur le train
cols_nommees = [c for c in cols if not c.startswith("mfcc")]
X_nom = ((va[cols_nommees] - X_tr[cols_nommees].mean()) / X_tr[cols_nommees].std()).values
print(f"\n=== Le clip le plus ressemblant vient-il de la même personne ? ({len(va)} clips, {va['spk'].nunique()} personnes, hasard ≈ {1 / va['spk'].nunique():.0%}) ===")
print(f"Caractéristiques nommées seules (hauteur, formants, stabilité...) : {taux_voisin(X_nom, False):.0%}")
print(f"Caractéristiques nommées + MFCC (timbre)                         : {taux_voisin(X_va, False):.0%}")
print(f"Empreinte ECAPA                                                  : {taux_voisin(E_va, True):.0%}")

# ---------- 4. Carte t-SNE ----------
mini = []
meta_mc = Path("data/mini_corpus/metadata.csv")
if meta_mc.exists():
    for _, r in pd.read_csv(meta_mc).iterrows():
        y = parole(r["fichier"])
        with torch.no_grad():
            e = model.encode_batch(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
        mini.append({"personne": r["personne"], "langue": r["langue"], "condition": r["condition"],
                     "emb": preparer(e, mu)[0]})
X = np.vstack([E_va] + ([np.stack([m["emb"] for m in mini])] if mini else []))
Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
Y_va, Y_mc = Y[:len(va)], Y[len(va):]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
ax = axes[0]
pts = ax.scatter(Y_va[:, 0], Y_va[:, 1], c=va["f0_hz"], cmap="Blues", vmin=60, s=18,
                 edgecolors=SURFACE, linewidths=0.5)
fig.colorbar(pts, ax=ax, fraction=0.04).set_label("Hauteur de la voix (Hz)", color=TEXTE_2)
ax.set_title("Common Voice : chaque îlot est une personne\n(couleur = hauteur de la voix)", loc="left", fontsize=10)

ax = axes[1]
ax.scatter(Y_va[:, 0], Y_va[:, 1], color=GRIS, s=12, label="Common Voice (validation)")
couleurs = [BLEU, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
marqueurs = {"fr": "o", "es": "s", "en": "^", "aucune": "X"}
personnes = sorted({m["personne"] for m in mini})
for i, p in enumerate(personnes):
    idx = [k for k, m in enumerate(mini) if m["personne"] == p]
    for k in idx:
        ax.scatter(*Y_mc[k], color=couleurs[i % 8], s=70, marker=marqueurs.get(mini[k]["langue"], "D"),
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
    centre = Y_mc[idx].mean(axis=0)
    ax.annotate(p, centre, xytext=(8, 8), textcoords="offset points", fontweight="bold", color=TEXTE)
legende = [Line2D([], [], color=GRIS, marker="o", linestyle="", label="Common Voice")]
legende += [Line2D([], [], color=couleurs[i % 8], marker="o", linestyle="", label=p) for i, p in enumerate(personnes)]
legende += [Line2D([], [], color=TEXTE_2, marker=mk, linestyle="", label=f"langue : {l}") for l, mk in marqueurs.items()]
ax.legend(handles=legende, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_title("Mini-corpus : une même personne reste groupée,\nquelle que soit la langue (forme = langue)", loc="left", fontsize=10)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
plt.savefig("results/tsne_ecapa.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nGraphiques : results/fisher_caracteristiques.png et results/tsne_ecapa.png")