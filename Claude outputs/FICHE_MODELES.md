# Fiche des modèles

Le sujet autorise les modèles pré-entraînés, à condition de documenter leur usage. Cette fiche liste tous les modèles utilisés ou entraînés, et ce qu'on en a fait.

## 1. Modèle pré-entraîné : ECAPA-TDNN (SpeechBrain)

| | |
|---|---|
| **Nom** | `speechbrain/spkrec-ecapa-voxceleb` |
| **Source** | Hugging Face Hub, téléchargé automatiquement dans `models/ecapa/` par `EncoderClassifier.from_hparams` |
| **Bibliothèque** | SpeechBrain **1.1.2** (PyTorch 2.15.0, torchaudio 2.11.1) |
| **Licence** | Apache-2.0 |
| **Architecture** | ECAPA-TDNN : convolutions 1D, blocs résiduels avec attention par canal (Squeeze-Excitation), pooling statistique attentif |
| **Entrée / sortie** | Audio 16 kHz mono → empreinte de **192 nombres** |
| **Entraînement d'origine** | VoxCeleb 1 et VoxCeleb 2 (données d'entraînement), loss AAM-softmax |
| **Performance annoncée** | EER 0,80 % sur VoxCeleb1-test (version nettoyée), selon la fiche du modèle |
| **Référence** | B. Desplanques, J. Thienpondt, K. Demuynck, *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*, Interspeech 2020 |

**Usage dans le projet**
- Le modèle est **gelé** : aucun de ses poids n'est modifié. On l'utilise seulement pour calculer des empreintes.
- Chaque signal est encodé en **un seul appel**, sans remplissage. C'est le cas dans la démo et dans le banc v3, ce qui garantit des empreintes identiques des deux côtés (`src/preparation.py`).
- Post-traitement : longueur ramenée à 1, soustraction de l'**empreinte moyenne** des clips de train cv26 (préparés comme la démo), puis longueur ramenée à 1 à nouveau.
- Comparaison par **cosinus** et décision par seuil, choisi sur la validation.
- Temps de calcul mesuré : environ **0,3 s pour 3 s d'audio** sur le CPU du PC portable (sans GPU).
- Cohérence vérifiée : un même signal donne la même empreinte sur Colab (GPU T4) et sur le PC (cosinus 1,00000).

**Pourquoi ce modèle**
- Sur les mêmes 710 paires de validation, il obtient **2,1 % d'EER**, contre 6,8 % pour la meilleure baseline.
- Il tourne en temps réel sur un CPU de portable.
- Il a été entraîné sur environ 7 000 personnes, en plusieurs langues et en conditions variées (vidéos YouTube). C'est ce qui lui permet de généraliser à des personnes inconnues.

**Limites connues**
- La fiche du modèle précise que ses auteurs ne garantissent pas ses performances sur d'autres données.
- Entraîné sur des **adultes** (célébrités) : les voix d'enfants risquent d'être moins bien reconnues.
- Sur des extraits très courts (3 s), l'empreinte dépend en partie des **sons prononcés**. Un changement de langue entre l'enrôlement et le test fait donc baisser le score.

## 2. Autres bibliothèques utilisées pour l'analyse

| Outil | Version | Usage |
|---|---|---|
| librosa | 0.11.0 | Chargement et rééchantillonnage, MFCC, détection des silences (`effects.split`, `top_db=30`), centroïde spectral, détection des attaques, pYIN (hauteur dans `build_bench_v2.py`) |
| praat-parselmouth | 0.4.7 | Praat depuis Python : hauteur (F0), formants F1 à F3, jitter, shimmer, HNR (`caracteristiques.py`), hauteur pour l'équilibrage cv27 |
| scikit-learn | 1.9.1 | Régression logistique, SVM, courbes ROC, t-SNE, calibration |
| streamlit | 1.67.0 | Interface de la démo |

## 3. Modèles entraînés dans le projet

| Modèle | Script | Données | Statut |
|---|---|---|---|
| Régression logistique et SVM sur paires MFCC | `baseline_classifiers.py` | `pairs_train.csv` (85 locuteurs cv26) | Baseline (non sauvegardés) |
| MLP 80 → 64 → 32 → 1 sur paires MFCC | `baseline_mlp.py` | `pairs_train.csv` | Baseline, `models/mlp_mfcc.pt` |
| Couche de metric learning **v1** : résiduelle 192 × 192, triplet loss | `metric_learning.py` | 85 locuteurs cv26 train, 6 030 empreintes | **Non retenue** (overfit), `models/couche_metric.pt` |
| Couche de metric learning **v2 « diagonale »** : un poids par dimension (192 paramètres), AAM-softmax | `metric_learning_v2.py` | 851 locuteurs cv27 (94 autres tenus à l'écart pour la surveillance) | **Non retenue** (pas de gain fiable), `models/couche_metric_v2_diagonale.pt` |
| Couche de metric learning **v2 « résiduelle rang 32 »** (12 480 paramètres), AAM-softmax | `metric_learning_v2.py` | idem | **Non retenue** (epoch gardée : 0), `models/couche_metric_v2_residuelle_r32.pt` |
| Calibration score → probabilité (régression logistique) | `eval_v3.py` | Paires de validation du banc v3 | **Utilisée dans la démo** : p = 1 / (1 + exp(−(10,54 × score − 3,38))) |

Les poids ne sont pas versionnés (`models/` est dans le `.gitignore`), mais chaque script les régénère.

## 4. Fichier de réglages de la démo : `models/reference_demo.npz`

Il est produit par `eval_v3.py`. L'ancienne version (banc v2) est gardée sous `models/reference_demo_v1.npz`.

| Clé | Contenu |
|---|---|
| `moyenne` | Empreinte moyenne à soustraire (192 nombres), source : « cv26 préparée démo » |
| `seuils` | `{"3": 0.416, "5": 0.447, "10": 0.482}` (1 % d'imposteurs acceptés, voix proches, validation) |
| `calib_a`, `calib_b` | 10,54 et −3,38 |
| `ecart_intra` | Écart habituel de chaque caractéristique nommée chez une même personne (pour l'explication) |
| `banc`, `moyenne_source` | `"v3"`, `"cv26 préparée démo"` (traçabilité) |

## 5. Technologies explorées et non retenues

| Technologie | Décision | Raison |
|---|---|---|
| **NVIDIA NeMo** (TitaNet) | Écartée | Orientée Linux et GPU, installation lourde sous Windows sans GPU |
| **WeSpeaker** | Non testée par manque de temps | Aurait permis de comparer une deuxième famille d'empreintes pré-entraînées |
| **Fine-tuning d'ECAPA** | Non réalisé | Avec quelques centaines de locuteurs, le risque d'overfit est encore plus fort que pour une couche légère, qui n'a déjà rien apporté |
| **AS-norm** (normalisation des scores par cohorte) | Écartée | Améliore un peu l'EER mais laisse passer des imposteurs à score élevé (refus à 1 % d'imposteurs : 12 % → 27,5 %) |
| **ASVspoof** (détection d'usurpation) | Non utilisé | Hors périmètre ; voir la section « Limites et attaques » du rapport |
