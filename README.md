# Voiceprint : ta voix, ton identité

Voiceprint est un système de vérification du locuteur à partir d'enregistrements vocaux. Il détermine si deux voix appartiennent à la même personne. Il est conçu pour fonctionner sur des personnes jamais entendues pendant le développement, quels que soient leur langue, les mots prononcés et leur genre. Une seule personne s'enrôle à la fois ; ensuite, autant de personnes que l'on veut peuvent se faire tester, l'une après l'autre, sans nouvel enrôlement.

Projet individuel réalisé par **Matéo Lebourdais** pendant la Piscine **IA Lab de l'ECE**.

> ⚠️ **Attention :** ceci est une démonstration pédagogique, pas un dispositif d'authentification sécurisé. Une empreinte vocale est une donnée biométrique : seule l'empreinte est stockée en local (jamais l'audio), et elle peut être supprimée à tout moment selon le souhait de la personne.

## En bref

| | |
|---|---|
| **Modèle retenu** | ECAPA-TDNN pré-entraîné (`speechbrain/spkrec-ecapa-voxceleb`, SpeechBrain 1.1.2) |
| **Calcul du score** | Similarité cosinus entre deux empreintes de 192 nombres, après soustraction de l'empreinte moyenne |
| **Enrôlement visé** | 3 secondes de parole (5 s et 10 s aussi proposées) |
| **Seuils** (au plus 1 % d'imposteurs acceptés, voix proches, vérifiés sur la validation cv26 et sur 290 personnes cv27) | 0,428 (3 s) · 0,447 (5 s) · 0,482 (10 s) |
| **Résultat en validation** (banc v3, enrôlement 3 s, voix proches, test jusqu'à 10 s) | EER 1,9 % · minDCF 0,172 · 5,3 % de vraies personnes refusées |
| **Résultat du test final** (29 personnes jamais utilisées, enrôlement 3 s, voix proches, test jusqu'à 10 s) | EER 3,0 % · AUC 0,997 · 0 % d'imposteurs acceptés · 11,6 % de vraies personnes refusées |
| **Comparaison baseline / ECAPA** (mêmes paires de validation v1) | Baseline MFCC + SVM : EER 6,8 % · ECAPA : EER 2,1 % |
| **Données** | Common Voice français 26.0 et 27.0, licence CC0-1.0 |

## Documentation

| Document | Contenu |
|---|---|
| [`docs/RAPPORT.md`](docs/RAPPORT.md) | Rapport technique : protocole, tableau comparatif, ablations, metric learning, explicabilité, grande validation, seuils et calibration, test final, cas d'échec, limites et attaques |
| [`docs/FICHE_CORPUS.md`](docs/FICHE_CORPUS.md) | Fiche des corpus : source, accès, licence, volume, composition, découpage, paires, risques de fuite, limites |

## Installation

**Prérequis**
- Python **3.11.9**
- **ffmpeg** dans le PATH (pour convertir les fichiers MP3)

**Commandes** (Windows, dans un terminal PowerShell)

```
git clone https://github.com/mateo-grp1/VoicePrint_Piscine.git
cd VoicePrint_Piscine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sous macOS / Linux, la ligne d'activation devient `source .venv/bin/activate`. Au premier lancement, le modèle ECAPA-TDNN (environ 80 Mo) est téléchargé automatiquement depuis Hugging Face.

## Données

Aucune donnée n'est versionnée : `data/` et `models/` sont dans le `.gitignore`, ainsi que tous les fichiers audio. La licence de Common Voice interdit de redistribuer le corpus, et les enregistrements du mini-corpus sont personnels.

1. **Common Voice Scripted Speech 26.0 – French of France** (`fr-FR`), sur [Mozilla Data Collective](https://datacollective.mozillafoundation.org/). Il faut accepter les conditions, télécharger l'archive, puis l'extraire dans `data/raw/cv26_fr/fr-FR/`.
2. **Common Voice Scripted Speech 27.0 – French** (28,55 Go) : ce corpus n'est pas téléchargé sur le PC. Le notebook [`notebooks/04_preparer_cv27_colab.ipynb`](notebooks/04_preparer_cv27_colab.ipynb) le traite sur Google Colab avec un GPU T4, puis ne rapatrie que les empreintes dans `data/cv27/` (environ 45 Mo). Il faut une clé API Mozilla Data Collective, à ranger dans les *Secrets* de Colab sous le nom `MDC_API_KEY`.
3. **Mini-corpus** (facultatif) : il s'enregistre avec `python src/enregistrer_corpus.py`. Le consentement est demandé avant tout enregistrement.

## Reproduire les expériences

Tous les scripts se lancent depuis la racine du projet. Chacun écrit ses résultats dans `results/` et ajoute ses lignes au tableau commun `results/resultats.csv`. Ils utilisent tous la même fonction d'évaluation (`src/evaluate.py`) : EER, minDCF, AUC, ROC et taux de vraies personnes refusées à 1 % d'imposteurs acceptés.

### 1. Données et paires (Common Voice 26.0)

```
python src/make_metadata.py        # data/metadata.csv : locuteurs avec au moins 5 clips
python src/convert_audio.py        # MP3 -> wav 16 kHz mono, retrait des clips < 2 s
python src/split_speakers.py       # découpage PAR LOCUTEUR 60/20/20 + test de fuite automatique
python src/make_pairs.py           # paires 50/50 pour train / val / test
```

### 2. Baseline classique (MFCC)

```
python src/features_mfcc.py         # 20 MFCC (moyenne + écart-type = 40 nombres par clip)
python src/baseline_cosine.py       # cosinus brut / normalisé
python src/baseline_classifiers.py  # régression logistique + SVM sur les paires
python src/baseline_mlp.py          # petit réseau PyTorch + courbes d'entraînement
```

### 3. Empreintes pré-entraînées (ECAPA-TDNN)

```
python src/ecapa_eval.py           # ECAPA + cosinus sur les mêmes paires que la baseline
python src/build_bench_v2.py       # banc v2 : voix proches (selon la hauteur fréquentielle) + extraits de 3 s
python src/ecapa_bench.py          # ECAPA par durée et par type d'imposteur
python src/reglages_v2.py          # ablation : soustraction de la moyenne + AS-norm
python src/enrolement_v2.py        # ablation : durée d'enrôlement 3/5/8/10 s + bloc ou morceaux
python src/bruit_v2.py             # robustesse : ambiance et brouhaha à 20/10/5 dB
```

### 4. Explicabilité et mini-corpus

```
python src/explicable.py           # ratio de Fisher + carte t-SNE
python src/eval_mini_corpus.py     # vérification "vraie vie" (si le mini-corpus existe)
```

### 5. Metric learning

```
python src/metric_learning.py      # v1 : triplet loss sur 85 locuteurs (non retenue : overfit)
python src/exporter_exclusions.py  # locuteurs val/test cv26 à exclure (à lancer AVANT Colab)
# Corpus v2 : exécuter notebooks/04_preparer_cv27_colab.ipynb sur Colab, puis copier les empreintes dans data/cv27/
python src/verifier_cv27.py        # anti-fuite par split + cohérence des empreintes Colab/PC
python src/metric_learning_v2.py   # v2 : AAM-softmax sur 945 locuteurs (non retenue : pas de gain fiable)
```

### 6. Banc v3 et réglages de la démo (configuration finale)

```
python src/build_bench_v3.py       # mêmes paires, voix préparées exactement comme dans la démo
python src/eval_v3.py              # choix de la moyenne + seuils + calibration
```

### 7. Grande validation (290 personnes Common Voice 27.0) et seuils finaux

```
python src/separer_cv27.py         # 290 personnes cv27 en validation, le reste en entraînement (anti-fuite)
# Exécuter notebooks/05_banc_cv27_colab.ipynb sur Colab, puis copier banc_cv27val.npz et banc_cv27val_info.csv dans data/cv27/
python src/eval_cv27val.py         # configuration actuelle mesurée sur 290 personnes (ne modifie pas la démo)
python src/figer_seuils.py         # seuils finaux : le plus prudent de toutes les validations -> configuration figée
```

### 8. Test final

```
python src/test_final.py           # évaluation UNIQUE sur les 29 personnes de test, configuration figée
python src/pires_paires.py         # liste les pires essais du test final, à écouter (cas d'échec)
```

## Démonstration

```
python src/build_bench_v3.py       # une seule fois
python src/eval_v3.py              # crée models/reference_demo.npz
python src/figer_seuils.py         # applique les seuils finaux (0,428 / 0,447 / 0,482)
streamlit run app.py
```

1. **Enrôlement** : on choisit la durée (3 / 5 / 10 s), on coche ensuite le consentement, puis on parle librement au micro **ou** on importe un fichier audio (wav, mp3, flac, ogg), par exemple un enregistrement fourni par l'organisation. Les silences et les pauses sont retirés, donc « 3 s » veut dire 3 s de voix réelle.
2. **Vérification** : chaque personne parle à son tour (micro ou fichier audio), jusqu'à 10 s de parole analysées. Le système affiche une décision oui ✅ / non ❌, avec une confiance calibrée si la personne est acceptée. Le score brut et le seuil sont aussi affichés. Autant de personnes que l'on veut peuvent se faire tester sans nouvel enrôlement.
3. **Suppression** : un bouton efface l'empreinte (`data/enrolement/empreinte.npz`).

## Organisation du dépôt

```
app.py                  Démonstration Streamlit
requirements.txt        Dépendances
src/                    Scripts Python
notebooks/              Exploration (01-03), préparation Colab de Common Voice 27.0 (04), grande validation (05)
results/                Tableaux et figures de résultats
docs/                   Rapport et fiche des corpus
data/   (hors GitHub)   Audio, paires, caches d'empreintes, mini-corpus, empreinte de la démo
models/ (hors GitHub)   ECAPA téléchargé, réglages de la démo, petits modèles entraînés
```

## Règles suivies

- **Séparation par locuteur** jamais par enregistrement. Les intersections sont vérifiées automatiquement, et aucun locuteur de validation ou de test cv26 ne se trouve dans les données d'entraînement cv27.
- **Seuils et calibration** choisis sur la validation, jamais sur le test. Le test final a été lancé une seule fois, sans aucun réglage ensuite.
- **Mêmes paires et même fonction d'évaluation** pour toutes les approches.
- **Minimisation** : l'audio de démo n'est jamais conservé, et l'archive de 28 Go de Common Voice 27.0 a été supprimée après extraction des empreintes.

## Licences et crédits

- Common Voice : CC0-1.0, Mozilla Foundation. <u>Il est interdit de chercher à identifier les locuteurs et de redistribuer le corpus.</u>
- ECAPA-TDNN : `speechbrain/spkrec-ecapa-voxceleb`, licence Apache-2.0, entraîné sur VoxCeleb 1 et 2. Référence : Desplanques et al., *ECAPA-TDNN*, Interspeech 2020.
- SpeechBrain : Ravanelli et al., 2021 (arXiv:2106.04624).
- Praat via `praat-parselmouth`.