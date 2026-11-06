# src/baseline_mlp.py
# Petit réseau de neurones (MLP) sur les paires : mêmes entrées que la LogReg.
# Entraîné sur pairs_train, suivi et évalué sur pairs_val.
# À lancer depuis la racine : python src/baseline_mlp.py
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate, enregistrer
from pair_features import charger_mfcc_normalises, entrees_paires

torch.manual_seed(42)                     # graine fixe : résultats reproductibles
EPOCHS, BATCH = 60, 64

# 1. Données : exactement les mêmes entrées que la LogReg
Xn, idx = charger_mfcc_normalises()
p_train = pd.read_csv("data/pairs/pairs_train.csv")
p_val = pd.read_csv("data/pairs/pairs_val.csv")
X_tr = entrees_paires(Xn, idx, p_train, "diff_prod")
X_va = entrees_paires(Xn, idx, p_val, "diff_prod")

scaler = StandardScaler().fit(X_tr)       # appris sur le TRAIN uniquement
X_tr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
X_va = torch.tensor(scaler.transform(X_va), dtype=torch.float32)
y_tr = torch.tensor(p_train["label"].values, dtype=torch.float32)
y_va = torch.tensor(p_val["label"].values, dtype=torch.float32)

# 2. Le réseau : 80 -> 64 -> 32 -> 1 score
modele = nn.Sequential(
    nn.Linear(X_tr.shape[1], 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
perte_fn = nn.BCEWithLogitsLoss()         # loss adaptée à une réponse 0/1
opt = torch.optim.Adam(modele.parameters(), lr=1e-3, weight_decay=1e-4)

# 3. Boucle d'entraînement
hist = {"loss_train": [], "loss_val": [], "eer_val": []}
meilleure = {"loss_val": float("inf"), "epoch": None, "poids": None}
debut = time.time()

for epoch in range(EPOCHS):
    modele.train()                        # mode entraînement (dropout actif)
    ordre = torch.randperm(len(X_tr))     # on mélange les paires à chaque epoch
    total = 0.0
    for i in range(0, len(X_tr), BATCH):
        b = ordre[i:i + BATCH]
        opt.zero_grad()                               # on efface les calculs précédents
        perte = perte_fn(modele(X_tr[b]).squeeze(1), y_tr[b])
        perte.backward()                              # calcule comment ajuster chaque poids
        opt.step()                                    # ajuste les poids
        total += perte.item() * len(b)

    modele.eval()                         # mode évaluation (dropout éteint)
    with torch.no_grad():
        s_va = modele(X_va).squeeze(1)
        loss_val = perte_fn(s_va, y_va).item()
    eer_val = evaluate(s_va.numpy(), p_val["label"])["eer"]

    hist["loss_train"].append(total / len(X_tr))
    hist["loss_val"].append(loss_val)
    hist["eer_val"].append(eer_val)

    if loss_val < meilleure["loss_val"]:  # on garde le meilleur état sur la validation
        meilleure = {"loss_val": loss_val, "epoch": epoch + 1,
                     "poids": copy.deepcopy(modele.state_dict())}

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch + 1:2d} | loss train {hist['loss_train'][-1]:.3f} "
              f"| loss val {loss_val:.3f} | EER val {eer_val:.3f}")

duree = time.time() - debut

# 4. Courbes d'entraînement (la "visualisation" demandée par le sujet)
Path("results").mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["loss_train"], label="train")
ax[0].plot(hist["loss_val"], label="validation")
ax[0].axvline(meilleure["epoch"] - 1, color="grey", linestyle="--", label="meilleure epoch")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["eer_val"], color="tab:green")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("EER"); ax[1].set_title("EER sur la validation")
plt.savefig("results/mlp_courbes.png", dpi=120, bbox_inches="tight")
plt.close()

# 5. Évaluation finale avec le meilleur état
modele.load_state_dict(meilleure["poids"])
modele.eval()
with torch.no_grad():
    s_tr = modele(X_tr).squeeze(1).numpy()
    s_va = modele(X_va).squeeze(1).numpy()
r_tr = evaluate(s_tr, p_train["label"])
r_va = evaluate(s_va, p_val["label"], nom="mlp_diff_prod")
enregistrer("MFCC40 + MLP |a-b| + a*b", "val", r_va,
            notes=f"meilleure epoch {meilleure['epoch']}/{EPOCHS}, {duree:.0f} s")

Path("models").mkdir(exist_ok=True)
torch.save(meilleure["poids"], "models/mlp_mfcc.pt")   # models/ est exclu de GitHub

print(f"\nMeilleure epoch : {meilleure['epoch']} | temps : {duree:.0f} s")
print(f"EER train {r_tr['eer']:.3f} | EER val {r_va['eer']:.3f} "
      f"| minDCF val {r_va['min_dcf']:.3f} | AUC val {r_va['auc']:.3f}")