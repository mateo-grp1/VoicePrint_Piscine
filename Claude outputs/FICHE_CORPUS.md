# Fiche des corpus

Cette fiche décrit toutes les données utilisées : leur source, leur accès, leur licence, leur volume, leur composition, leur découpage, les paires construites, les risques de fuite et les limites.
**Aucune donnée n'est redistribuée** : `data/` est exclu du dépôt.

## Vue d'ensemble

| Corpus | Rôle dans le projet | Locuteurs utilisés | Licence | Dans le dépôt ? |
|---|---|---|---|---|
| Common Voice 26.0, français de France (`fr-FR`) | Baseline, évaluation de toutes les approches (train / validation / test) | 142 | CC0-1.0 | Non |
| Common Voice 27.0, français (sous-ensemble) | Entraînement de la couche de metric learning v2 uniquement | 945 | CC0-1.0 | Non (seules les empreintes, sur le PC) |
| Mini-corpus « vraie vie » | Vérification qualitative (2 personnes, 3 langues, distance au micro) | 2 | Consentement oral, usage interne | Non |
| VoxCeleb 1 et 2 | Utilisé **indirectement** : c'est le corpus d'entraînement d'ECAPA | — | Voir la fiche des modèles | Non |
| Common Voice Spontaneous Speech 5.0, français | Envisagé pour un test « parole spontanée », **abandonné** | — | CC0-1.0 | Non |

---

## 1. Common Voice 26.0, français de France

**Source et accès**
- *Common Voice Scripted Speech 26.0 – French of France*, Mozilla Common Voice, distribué par Mozilla Data Collective.
- On crée un compte, on accepte les conditions sur la page du dataset, puis on télécharge l'archive `fr-FR.tar.gz` et on l'extrait dans `data/raw/cv26_fr/fr-FR/`.
- Licence **CC0-1.0**. Conditions : **interdit de chercher à identifier les locuteurs** et **interdit de redistribuer** le corpus.

**Nature des données**
- Phrases **lues** (corpus de reconnaissance de la parole), enregistrées par des volontaires avec leur propre micro, chez eux. Les conditions sont donc très variables : qualité du micro, bruit de fond, réverbération.
- Format d'origine : MP3. Converti en wav 16 kHz mono.
- Métadonnées par clip : `client_id` (identifiant haché du locuteur), `path`, `sentence`, `age`, `gender` (déclaratifs et facultatifs).

**Filtrage** (`make_metadata.py`, `convert_audio.py`)

| Étape | Règle | Résultat |
|---|---|---|
| Archive brute | `train.tsv` + `dev.tsv` + `test.tsv` | 5 490 clips, 206 locuteurs |
| Locuteurs | au moins 5 clips | — |
| Clips | au moins 2,0 s après conversion | — |
| **Corpus utilisé** | | **5 282 clips, 142 locuteurs, 7,45 h** |

- Durée des clips : moyenne 5,1 s, médiane 5,0 s, de 2,0 à 13,7 s.
- `client_id` remplacé par un code court (`cv_000`…). La correspondance n'est pas publiée.

**Découpage par locuteur** (`split_speakers.py`, graine 42, 60/20/20)

On n'a **pas** gardé le découpage officiel : on a refait un découpage **par locuteur**, avec un test automatique qui vérifie qu'aucun locuteur n'apparaît dans deux groupes.

| Groupe | Locuteurs | Clips | Heures | Usage |
|---|---|---|---|---|
| train | 85 | 3 361 | 4,79 | Apprentissage (baseline), moyenne à soustraire, voix de brouhaha |
| validation | 28 | 601 | 0,84 | **Tous les choix** : seuils, calibration, réglages, choix d'epoch |
| test | 29 | 1 320 | 1,82 | **Fermé** jusqu'au test final |

**Déséquilibres documentés**
- **Contributeurs très inégaux** : de 5 à 505 clips par locuteur (médiane 8). Les 5 plus gros contributeurs font **34 %** des clips. On a donc plafonné le nombre de paires par locuteur.
- **Genre** : déclaré par une minorité de locuteurs, et majoritairement masculin parmi ceux qui l'ont déclaré. Les « voix proches » ont donc été définies par la **hauteur mesurée** (F0) et non par le genre déclaré.
- **Langue** : uniquement du français lu. Les autres langues ne sont testées que sur le mini-corpus.
- **Hauteur de voix des 28 locuteurs de validation** : médiane 130 Hz, de 96 à 252 Hz.

**Paires construites**

| Fichier | Construction | Volume |
|---|---|---|
| `pairs_train.csv` | Positives : jusqu'à 50 par locuteur. Négatives : 2 locuteurs tirés au hasard. 50/50 | 5 394 |
| `pairs_val.csv` | Idem, jusqu'à 20 positives par locuteur | 710 (355 / 355) |
| `pairs_test.csv` | Idem, jusqu'à 20 positives par locuteur | 940 (470 / 470) |
| `pairs_val_v2.csv` (banc v2) | 320 positives (jusqu'à 15 par locuteur), 320 négatives au hasard, 320 négatives **« voix proches »** (le quart des couples de locuteurs dont la hauteur est la plus proche : écart de 0 à 14 Hz). Chaque paire existe en 3 versions : clip **entier**, extrait de **3 s**, extrait de **2 s** (pris au milieu du clip brut) | 960 × 3 |
| `pairs_val_v3.csv` (banc v3) | Les **mêmes 960 paires**, préparées **exactement comme dans la démo** (silences retirés, enrôlement de 3, 5 ou 10 s de parole, test de 3 s ou jusqu'à 10 s, bruit ajouté avant le retrait des silences) | 960 |

Vérifications automatiques dans `make_pairs.py` : tous les locuteurs d'une paire appartiennent au bon groupe, les étiquettes sont cohérentes, et aucun clip n'est comparé à lui-même.

---

## 2. Common Voice 27.0, français (sous-ensemble d'entraînement)

**Pourquoi ce corpus :** la première version du metric learning (85 locuteurs) a sur-appris. Il fallait **beaucoup plus de personnes différentes**, et non plus de clips.

**Source et accès**
- *Common Voice Scripted Speech 27.0 – French*, version `cv-corpus-27.0-2026-09-11`, publiée le 17/09/2026.
- Identifiant Mozilla Data Collective : `cmu5kub4600pkmh078n7n5ogl`. Archive de 28,55 Go.
- Licence **CC0-1.0**, mêmes interdictions que la version 26.0.
- Description officielle : 869 961 clips, 1 217 h, 21 116 locuteurs, avec de nombreux accents (France, Canada, Belgique, Suisse, Afrique…).

**Traitement** (notebook `04_preparer_cv27_colab.ipynb`, exécuté sur Google Colab avec un GPU T4)

| Étape | Résultat |
|---|---|
| Clips validés | 792 011 clips, 20 051 locuteurs |
| Retrait des clips de moins de 3,8 s (`clip_durations.tsv`) | 600 858 clips |
| **Exclusion anti-fuite** des locuteurs de validation et de test cv26 | 57 locuteurs retrouvés et exclus, par **deux méthodes** : 57/57 `client_id` et 1 921/1 921 noms de clips |
| Locuteurs avec au moins 12 phrases différentes | 4 912 |
| Tirage aléatoire (graine 42) | 3 000 candidats, 15 clips chacun |
| Hauteur médiane mesurée (Praat, 2 clips par candidat) | 2 169 voix graves (< 165 Hz), 831 voix aiguës |
| **Équilibrage par hauteur** | 500 voix graves + 500 voix aiguës |
| Au moins 3 s de parole par clip, 8 à 12 clips par locuteur | **945 locuteurs** (462 graves, 483 aigus), **10 353 clips**, parole médiane 4,1 s |
| Empreintes ECAPA : 6 versions par clip | 62 118 empreintes |

Les 6 versions de chaque clip sont : 2 propres, ambiance à 20 et 10 dB, brouhaha à 20 et 10 dB. Chacune utilise un extrait de 3 s de parole pris au hasard.

**Minimisation**
- On ne conserve ni l'âge, ni le genre, ni l'accent, ni le texte des phrases. On garde seulement `locuteur`, `client_id`, `clip`, `duree_parole` et `f0_locuteur`.
- L'archive de 28 Go a été supprimée après traitement. L'audio sélectionné (451 Mo) n'est conservé que dans un Google Drive **privé**, pour pouvoir reproduire le calcul. Le PC ne contient que les empreintes.

**Vérifications sur le PC** (`verifier_cv27.py`)
- Par groupe cv26 : **0** locuteur commun avec la validation et **0** avec le test. 5 locuteurs sont communs avec le train cv26, ce qui est autorisé (ce sont des personnes d'entraînement).
- Le même signal synthétique donne exactement la même empreinte sur Colab (GPU) et sur le PC (CPU) : **cosinus 1,00000**.

---

## 3. Mini-corpus « vraie vie »

**But :** tester ce que Common Voice ne contient pas : **changement de langue**, **fausse langue** (des sons sans vrais mots), **distance au micro**, avec le micro intégré du PC utilisé pour la démo.

| | |
|---|---|
| Personnes | 2 : `p00` (l'auteur, trilingue français / espagnol / anglais) et `p01` (un proche, français) |
| Consentement | Texte lu à la personne avant tout enregistrement ; refus = rien n'est enregistré (`enregistrer_corpus.py`) |
| Protocole | Enrôlement de 10 s, puis tests de 5 s : 2 par langue, 1 en fausse langue, 1 à environ 1,5 m du micro |
| Volume | 14 fichiers (p00 : 1 enrôlement + 8 tests ; p01 : 1 enrôlement + 4 tests) |
| Format | wav 16 kHz mono, micro intégré du PC, enregistré le 12/11/2026 |
| Minimisation | Codes `p00` / `p01` au lieu des noms ; **genre non précisé** ; fichiers locaux, jamais publiés, supprimés à la fin du projet ou sur demande |

**Limite :** 2 personnes ne suffisent pas pour faire des statistiques. Ce corpus sert seulement d'**illustration qualitative** (et à la carte t-SNE). Les chiffres viennent de Common Voice.

---

## 4. Corpus envisagés et non retenus

| Corpus | Raison |
|---|---|
| **VoxCeleb** (téléchargement direct) | Inutile de l'entraîner nous-mêmes : ECAPA a déjà été entraîné dessus. Il est utilisé à travers le modèle pré-entraîné. |
| **Common Voice dans d'autres langues** (pour l'entraînement) | Chaque locuteur Common Voice ne parle qu'**une** langue. En entraîner une couche en multilingue lui apprendrait le raccourci « langue différente = personne différente », l'inverse de ce que veut la démo. |
| **Common Voice Spontaneous Speech 5.0, français** (52 locuteurs) | Trop petit pour entraîner. Envisagé pour un test « parole spontanée », puis abandonné pour se concentrer sur le corpus 27.0. |
| **ASVspoof** | Détection d'usurpation : hors du périmètre du projet. Cité dans les limites. |

---

## 5. Risques de fuite et parades

| Risque | Parade |
|---|---|
| Un même locuteur dans deux groupes | Découpage **par locuteur** et test automatique d'intersection vide (`split_speakers.py`, `make_pairs.py`) |
| Locuteurs cv26 de validation ou de test présents dans cv27 | Exclusion par `client_id` **et** par noms de clips, vérification par groupe sur le PC (`exporter_exclusions.py`, notebook 04, `verifier_cv27.py`) |
| Réglages faits sur les données de test | Tous les choix sont faits sur la validation ; le test reste fermé jusqu'au test final |
| Choix d'epoch sur la validation qui gonfle les chiffres de validation | Assumé et signalé ; le test final donnera le chiffre non biaisé |
| Un clip de test qui sert aussi à l'enrôlement (banc v3) | Pour les paires positives, le clip de test est exclu du bloc d'enrôlement |

## 6. Limites générales des données

- **Français lu uniquement** pour toutes les mesures chiffrées : la démo, elle, utilisera de la parole **spontanée**, dans **plusieurs langues**.
- **Pas de voix d'enfants identifiées** : l'âge déclaré le plus jeune dans Common Voice est « adolescent ». La reconnaissance des voix d'enfants n'est donc pas mesurée, et ECAPA a lui aussi été entraîné sur des adultes.
- **Micro et pièce inconnus** dans Common Voice, mais variés. La salle de la démo ne peut pas être reproduite à l'avance.
- **Taille du banc de validation** : 320 paires positives et 320 voix proches. L'incertitude est d'environ ±1 point sur les taux d'erreur, et le seuil à 1 % d'imposteurs ne repose que sur environ 3 paires d'imposteurs.
- **Bruit simulé** (bruit rose, brouhaha de voix Common Voice) et non enregistré en conditions réelles.
