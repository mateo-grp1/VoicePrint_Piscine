# Voiceprint : ta voix, ton identité

Système de **vérification du locuteur** : à partir de deux enregistrements, il décide s'ils viennent de la même personne. Il est conçu pour **généraliser à des personnes jamais entendues** pendant le développement, quelles que soient leur langue, leurs mots et leur genre.

Projet individuel réalisé pendant la Piscine IA Lab.

> ⚠️ **Démonstration pédagogique, pas un dispositif d'authentification sécurisé.** Le système ne détecte ni un enregistrement rejoué ni une voix clonée. Une empreinte vocale est une **donnée biométrique** : seule l'empreinte est stockée (jamais l'audio), en local, et elle peut être supprimée à tout moment.

---

## En bref

| | |
|---|---|
| **Modèle retenu** | ECAPA-TDNN pré-entraîné (`speechbrain/spkrec-ecapa-voxceleb`, SpeechBrain 1.1.2), utilisé gelé |
| **Score** | Similarité cosinus entre deux empreintes de 192 nombres, après soustraction de l'empreinte moyenne |
| **Enrôlement visé** | **3 secondes de parole** (5 s et 10 s aussi proposées) |
| **Seuils** (1 % d'imposteurs acceptés, voix proches, validation) | **0,416** (3 s) · 0,447 (5 s) · 0,482 (10 s) |
| **Résultat principal** (validation, banc v3, enrôlement 3 s, voix proches, test jusqu'à 10 s) | **EER 1,9 %** · minDCF 0,172 · 5,3 % de vraies personnes refusées au seuil |
| **Meilleure baseline classique** (MFCC + SVM, paires de validation v1) | EER 6,8 % (contre 2,1 % pour ECAPA sur les mêmes paires) |
| **Données** | Common Voice français 26.0 (évaluation) et 27.0 (entraînement du metric learning), licence CC0-1.0 |

Le détail des résultats, des ablations et des limites se trouve dans [`docs/RAPPORT.md`](docs/RAPPORT.md).

---

## Documentation

| Document | Contenu |
|---|---|
| [`docs/RAPPORT.md`](docs/RAPPORT.md) | Rapport technique : protocole, tableau comparatif, ablations, explicabilité, calibration, cas d'échec, limites et attaques |
| [`docs/FICHE_CORPUS.md`](docs/FICHE_CORPUS.md) | Fiche des corpus : source, accès, licence, volume, composition, découpage, paires, risques de fuite, limites |
| [`docs/FICHE_MODELES.md`](docs/FICHE_MODELES.md) | Modèles pré-entraînés utilisés (nom, version, source, licence) et modèles entraînés dans le projet |
| [`docs/JOURNAL.md`](docs/JOURNAL.md) | Journal de bord détaillé, étape par étape, avec ce que fait chaque script |

---

## Installation

**Prérequis**
- Python **3.11** (développé avec 3.11.9 sous Windows 11, sur CPU, sans carte graphique)
- **ffmpeg** dans le PATH (seulement pour convertir les MP3 de Common Voice : `src/convert_audio.py`)
- Un micro (seulement pour la démo et le mini-corpus)

**Environnement**

```powershell
git clone <URL_DU_DEPOT>          # [À COMPLÉTER]
cd VoicePrint_Piscine
python -m venv .venv
.venv\Scripts\activate             # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Sous PowerShell, si l'activation est refusée : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis on réessaie.

Le modèle ECAPA (environ 80 Mo) est téléchargé automatiquement depuis Hugging Face au premier lancement, dans `models/ecapa/`. Sous Windows, SpeechBrain affiche un avertissement sur les « symlinks » : il est sans conséquence.

---

## Données

Aucune donnée n'est versionnée : `data/` et `models/` sont dans le `.gitignore`, ainsi que tous les fichiers audio. La licence de Common Voice interdit de redistribuer le corpus, et les enregistrements du mini-corpus sont personnels.

1. **Common Voice Scripted Speech 26.0 – French of France** (`fr-FR`), sur [Mozilla Data Collective](https://datacollective.mozillafoundation.org). Il faut accepter les conditions, télécharger l'archive, puis l'extraire dans `data/raw/cv26_fr/fr-FR/` (qui contient `clips/`, `train.tsv`, `dev.tsv` et `test.tsv`).
2. **Common Voice Scripted Speech 27.0 – French** (28,55 Go) : ce corpus n'est **pas** téléchargé sur le PC. Le notebook [`notebooks/04_preparer_cv27_colab.ipynb`](notebooks/04_preparer_cv27_colab.ipynb) le traite sur Google Colab (GPU T4) et ne rapatrie que les empreintes (`data/cv27/`, environ 45 Mo). Il faut une clé API Mozilla Data Collective, à ranger dans les *Secrets* de Colab sous le nom `MDC_API_KEY`.
3. **Mini-corpus** (facultatif) : il s'enregistre avec `python src/enregistrer_corpus.py`. Le consentement est demandé avant tout enregistrement.

Le détail se trouve dans [`docs/FICHE_CORPUS.md`](docs/FICHE_CORPUS.md).

---

## Reproduire les expériences

Tous les scripts se lancent **depuis la racine du projet**. Chacun écrit ses résultats dans `results/` et ajoute ses lignes au tableau commun `results/resultats.csv`. Ils utilisent tous **la même fonction d'évaluation** (`src/evaluate.py` : EER, minDCF, AUC, ROC, taux de refus à 1 % d'imposteurs).

### 1. Données et paires (Common Voice 26.0)

```powershell
python src/make_metadata.py      # data/metadata.csv : locuteurs avec au moins 5 clips
python src/convert_audio.py      # MP3 -> wav 16 kHz mono, retrait des clips < 2 s
python src/split_speakers.py     # découpage PAR LOCUTEUR 60/20/20 (graine 42) + test de fuite automatique
python src/make_pairs.py         # paires 50/50 pour train / val / test
```

### 2. Baseline classique (MFCC)

```powershell
python src/features_mfcc.py          # 20 MFCC -> moyenne + écart-type = 40 nombres par clip
python src/baseline_cosine.py        # cosinus brut / normalisé (ablation de la normalisation)
python src/baseline_classifiers.py   # régression logistique et SVM sur les paires
python src/baseline_mlp.py           # petit réseau PyTorch + courbes d'entraînement (results/mlp_courbes.png)
```

### 3. Empreintes pré-entraînées (ECAPA-TDNN)

```powershell
python src/ecapa_eval.py         # ECAPA + cosinus, sur les MÊMES paires que la baseline
python src/build_bench_v2.py     # banc v2 : voix proches (hauteur) + extraits de 2 s et 3 s
python src/escapa_bench.py       # ECAPA par durée et par type d'imposteur
python src/reglages_v2.py        # ablation : soustraction de la moyenne, AS-norm
python src/enrolement_v2.py      # ablation : durée d'enrôlement 3/5/8/10 s, bloc ou morceaux
python src/bruit_v2.py           # robustesse : ambiance et brouhaha à 20/10/5 dB (environ 15 à 20 min)
```

### 4. Explicabilité et mini-corpus

```powershell
python src/explicable.py         # ratio de Fisher, plus proche voisin, carte t-SNE
python src/eval_mini_corpus.py   # vérification "vraie vie" (si le mini-corpus existe)
```

### 5. Metric learning

```powershell
python src/metric_learning.py        # v1 : triplet loss sur 85 locuteurs (non retenue : overfit)
# Corpus v2 : exécuter notebooks/04_preparer_cv27_colab.ipynb sur Colab, puis copier dans data/cv27/
python src/exporter_exclusions.py    # (avant Colab) locuteurs val/test cv26 à exclure
python src/verifier_cv27.py          # anti-fuite par split + cohérence des empreintes Colab/PC
python src/metric_learning_v2.py     # v2 : AAM-softmax sur 945 locuteurs (non retenue : pas de gain fiable)
```

### 6. Banc v3 et réglages de la démo (configuration finale)

```powershell
python src/build_bench_v3.py     # mêmes paires, voix préparées EXACTEMENT comme dans la démo (environ 40 min sur CPU)
python src/eval_v3.py            # choix de la moyenne, seuils, calibration -> models/reference_demo.npz
```

### 7. Test final

[À COMPLÉTER] Évaluation unique sur les paires de **test** (29 locuteurs jamais utilisés), avec la configuration figée.

---

## Démonstration

```powershell
python src/build_bench_v3.py     # une seule fois (cache)
python src/eval_v3.py            # crée models/reference_demo.npz
streamlit run app.py
```

1. **Enrôlement** : on choisit la durée (3 s par défaut), on coche le consentement, puis on parle librement. Les silences et les pauses sont retirés : « 3 s » veut dire 3 s de voix réelle.
2. **Vérification** : chaque personne parle à son tour, jusqu'à 10 s de parole analysées. Le système affiche une décision **✅ / ❌**, avec une **confiance calibrée** si la personne est acceptée, ainsi que le score brut et le seuil. Les passages s'enchaînent sans se réenregistrer.
3. **Pourquoi ?** : les 3 caractéristiques nommées (hauteur, formants, clarté…) qui diffèrent le plus de la voix enrôlée.
4. **Suppression** : un bouton efface l'empreinte (`data/enrolement/empreinte.npz`).

La voix est préparée par [`src/preparation.py`](src/preparation.py), **la même recette** que celle du banc d'essai v3. Les seuils affichés correspondent donc exactement aux conditions de la démo.

---

## Organisation du dépôt

```
app.py                      Démo Streamlit
requirements.txt            Dépendances (versions figées)
src/                        Scripts (voir le tableau ci-dessous)
notebooks/                  Exploration (01-03) et préparation Colab de Common Voice 27.0 (04)
results/                    Tableaux et figures (versionnés)
docs/                       Rapport, fiches et journal
data/   (hors GitHub)       Audio, paires, caches d'empreintes, mini-corpus, empreinte de la démo
models/ (hors GitHub)       ECAPA téléchargé, réglages de la démo, petits modèles entraînés
```

| Rôle | Scripts |
|---|---|
| Recette commune et outils partagés | `preparation.py`, `evaluate.py`, `normalisation.py`, `augmentation.py`, `caracteristiques.py`, `pair_features.py` |
| Données et paires | `make_metadata.py`, `convert_audio.py`, `split_speakers.py`, `make_pairs.py`, `build_bench_v2.py`, `build_bench_v3.py` |
| Baseline | `features_mfcc.py`, `baseline_cosine.py`, `baseline_classifiers.py`, `baseline_mlp.py` |
| ECAPA et ablations | `ecapa_eval.py`, `escapa_bench.py`, `reglages_v2.py`, `enrolement_v2.py`, `bruit_v2.py`, `eval_v3.py` |
| Explicabilité | `explicable.py`, `caracteristiques.py` |
| Metric learning | `metric_learning.py`, `exporter_exclusions.py`, `verifier_cv27.py`, `metric_learning_v2.py` |
| Mini-corpus | `enregistrer_corpus.py`, `eval_mini_corpus.py` |
| Historique (remplacé par `eval_v3.py`) | `calibration.py` |

---

## Règles suivies

- **Séparation par locuteur**, jamais par enregistrement. Les intersections sont vérifiées automatiquement, et aucun locuteur de validation ou de test cv26 ne se trouve dans les données d'entraînement cv27 (double vérification par `client_id` et par noms de clips).
- **Seuils et calibration choisis sur la validation**, jamais sur le test.
- **Mêmes paires et même fonction d'évaluation** pour toutes les approches.
- **Minimisation** : l'audio de démo n'est jamais conservé, le genre n'est pas collecté dans le mini-corpus, et l'archive de 28 Go de Common Voice 27.0 a été supprimée après extraction des empreintes.

## Licences et crédits

- Code du projet : [À COMPLÉTER : licence choisie, par exemple MIT]
- Common Voice : CC0-1.0, Mozilla Foundation. Il est interdit de chercher à identifier les locuteurs et de redistribuer le corpus.
- ECAPA-TDNN : `speechbrain/spkrec-ecapa-voxceleb`, licence Apache-2.0, entraîné sur VoxCeleb 1 et 2. Référence : Desplanques et al., *ECAPA-TDNN*, Interspeech 2020.
- SpeechBrain : Ravanelli et al., 2021 (arXiv:2106.04624).
- Praat via `praat-parselmouth` (mesures acoustiques).
