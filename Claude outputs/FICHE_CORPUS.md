# Fiche des corpus

Cette fiche décrit toutes les données utilisées et leurs informations : source, accès, licence, volume, composition, découpage, paires construites, risques de fuite et limites. **Aucune donnée n'est redistribuée** : `data/` est exclu du dépôt.

## Vue d'ensemble

| Corpus | Rôle dans le projet | Locuteurs utilisés | Licence | Dans le dépôt ? |
|---|---|---|---|---|
| Common Voice 26.0, français de France (fr-FR) | Baseline, évaluation de toutes les approches (train / validation / test) | 142 | CC0-1.0 | Non |
| Common Voice 27.0, français (sous-ensemble) | Entraînement du metric learning (655 personnes) et grande validation (290 personnes) | 945 | CC0-1.0 | Non (seules les empreintes, sur le PC) |
| Mini-corpus | Vérification qualitative | 2 | Consentement, usage interne | Non |
| VoxCeleb 1 et 2 | Corpus d'entraînement d'ECAPA | — | — | Non |

## 1. Common Voice 26.0, français de France

**Source et accès**

- *Common Voice Scripted Speech 26.0 – French of France*, Mozilla Common Voice, distribué par Mozilla Data Collective.
- On crée un compte, on accepte les conditions sur la page du dataset, puis on télécharge l'archive `fr-FR.tar.gz` et on l'extrait dans `data/raw/cv26_fr/fr-FR/`.
- Licence **CC0-1.0**. Conditions : **interdit de chercher à identifier les locuteurs** et **interdit de redistribuer** le corpus.

**Nature des données**

- Phrases lues, enregistrées individuellement par des volontaires avec leur propre micro. Les conditions sont donc très variables : qualité du micro, bruit de fond, réverbération...
- Format d'origine : MP3. Converti en wav 16 kHz mono.
- Métadonnées par clip : `client_id`, `path`, `sentence`, `age`, `gender`.

**Filtrage**

| Étape | Règle | Résultat |
|---|---|---|
| Archive brute | train.tsv + dev.tsv + test.tsv | 5 490 clips, 206 locuteurs |
| Locuteurs | au moins 5 clips | — |
| Clips | au moins 2.0 s après conversion | — |
| **Corpus utilisé** | | **5 282 clips, 142 locuteurs, 7.45 h** |

- Durée des clips : moyenne 5.1 s / médiane 5.0 s, de 2.0 à 13.7 s.
- `client_id` remplacé par un code court (`cv_000`...).

**Découpage par locuteur (60/20/20)**

On n'a pas gardé le découpage officiel : on a refait un découpage par locuteur, avec un test automatique qui vérifie qu'aucun locuteur n'apparaît dans deux groupes.

| Groupe | Locuteurs | Clips | Heures | Usage |
|---|---|---|---|---|
| train | 85 | 3 361 | 4.79 | Apprentissage (baseline), moyenne à soustraire, voix avec du bruit |
| validation | 28 | 601 | 0.84 | Tous les choix : seuils, calibration, réglages, choix d'epoch |
| test | 29 | 1 320 | 1.82 | Fermé jusqu'au test final |

**Déséquilibres documentés**

- **Contributeurs très inégaux** : de 5 à 505 clips par locuteur. Les 5 plus gros contributeurs font 34 % des clips. On a donc plafonné le nombre de paires par locuteur.
- **Genre** : déclaré par une minorité de locuteurs, et majoritairement masculin parmi ceux qui l'ont déclaré. Les « voix proches » ont donc été définies par la hauteur de voix mesurée (F0) et non par le genre déclaré.
- **Langue** : uniquement du français lu.
- **Hauteur de voix** des 28 locuteurs de validation : médiane 130 Hz, de 96 à 252 Hz.

**Paires construites**

| Fichier | Construction | Volume |
|---|---|---|
| `pairs_train.csv` | Positives : jusqu'à 50 par locuteur. Négatives : 2 locuteurs tirés au hasard. 50/50 | 5 394 |
| `pairs_val.csv` | Idem, jusqu'à 20 positives par locuteur | 710 |
| `pairs_test.csv` | Idem, jusqu'à 20 positives par locuteur | 940 |
| `pairs_val_v2.csv` (banc v2) | 320 positives (jusqu'à 15 par locuteur), 320 négatives au hasard, 320 négatives « voix proches ». Chaque paire existe en 3 versions : clip entier, extrait de 3 s et extrait de 2 s. | 960 × 3 |
| `pairs_val_v3.csv` (banc v3) | Les mêmes 960 paires, préparées comme dans l'application, pour que les mesures correspondent aux conditions réelles : silences retirés, enrôlement de 3, 5 ou 10 s de parole, test de 3 s ou jusqu'à 10 s, bruit ajouté avant le retrait des silences. | 960 |
| Banc cv27 (grande validation) | 290 personnes cv27, préparées comme dans l'application (notebook 05). 4 clips de test par personne, les autres forment l'enrôlement (3, 5 ou 10 s). Chaque enrôlement est comparé à tous les clips de test : 1 160 essais de la vraie personne, 83 808 essais d'imposteurs proches. | 1 160 + 83 808 |
| Test final (cv26, 29 personnes) | Même protocole : chaque personne s'enrôle une fois, jusqu'à 10 de ses autres clips servent de tests. Évalué une seule fois, avec la configuration figée. | 198 + 5 544 (dont 1 543 voix proches) |

## 2. Common Voice 27.0, français (sous-ensemble d'entraînement)

**Pourquoi ce corpus :** la première version du metric learning (85 locuteurs) a sur-appris. Il fallait beaucoup plus de personnes différentes, et non plus de clips.

**Source et accès**

- *Common Voice Scripted Speech 27.0 – French*, version `cv-corpus-27.0-2026-09-11`, publiée le 17/09/2026.
- Identifiant Mozilla Data Collective : `cmu5kub4600pkmh078n7n5ogl`. Archive de 28.55 Go.
- Licence **CC0-1.0**, mêmes interdictions que la version 26.0.
- Description officielle : 869 961 clips, 1 217 h, 21 116 locuteurs, avec de nombreux accents (France, Canada, Belgique, Suisse, Afrique...).

Comme très volumineuse son traitement c'est fait sur Google Colab avec un GPU T4 via le notebook `04_preparer_cv27_colab.ipynb`.

| Étape | Résultat |
|---|---|
| Clips validés | 792 011 clips, 20 051 locuteurs |
| Retrait des clips de moins de 3.8 s (`clip_durations.tsv`) | 600 858 clips |
| **Exclusion anti-fuite** des locuteurs de validation et de test cv26 | 57 locuteurs retrouvés et exclus |
| Locuteurs avec au moins 12 phrases différentes | 4 912 |
| Tirage aléatoire | 3 000 candidats, 15 clips chacun |
| Hauteur médiane mesurée (Praat, 2 clips par candidat) | 2 169 voix graves, 831 voix aiguës |
| Équilibrage par hauteur | 500 voix graves + 500 voix aiguës |
| Au moins 3 s de parole par clip, 8 à 12 clips par locuteur | 945 locuteurs, 10 353 clips, parole médiane 4.1 s |
| Empreintes ECAPA : 6 versions par clip | 62 118 empreintes |
| Séparation par locuteur (graine 42, `separer_cv27.py`) | 655 personnes d'entraînement, 290 de validation (145 voix graves + 145 aiguës). Les personnes déjà présentes dans le train cv26 sont placées d'office en entraînement. |

Les 6 versions de chaque clip sont : 2 propres, ambiance à 20 et 10 dB, brouhaha à 20 et 10 dB. Chacune utilise un extrait de 3 s de parole pris au hasard. Le bruit apprend à la couche à reconnaître une voix même dans une salle bruyante.

**Minimisation**

- On ne conserve ni l'âge, ni le genre, ni l'accent, ni le texte des phrases. On garde seulement `locuteur`, `client_id`, `clip`, `duree_parole` et `f0_locuteur`.
- L'archive de 28 Go a été supprimée après traitement. L'audio sélectionné (451 Mo) n'est conservé que dans un Google Drive privé, pour pouvoir reproduire le calcul. Le PC ne contient que les empreintes.

**Vérifications sur le PC** (`verifier_cv27.py`)

- Par groupe cv26 : 0 locuteur commun avec la validation et 0 avec le test.
- Le même signal synthétique donne exactement la même empreinte sur Colab (GPU) et sur le PC (CPU) : cosinus 1.00000.
- Grande validation (notebook 05) : même fichier de préparation de la voix sur Colab et sur le PC (vérifié automatiquement), même empreinte de contrôle (cosinus 1.00000).

## 3. Mini-corpus

Le but était de tester ce que Common Voice ne contient pas : changement de langue, sons sans vrais mots, distance au micro.

| | |
|---|---|
| Personnes | 2 (FR / EN / ES) |
| Consentement | Texte lu à la personne avant tout enregistrement si refus rien n'est enregistré |
| Protocole | Enrôlement de 10 s, puis tests de 5 s : 2 par langue, 1 à environ 1.5 m du micro |
| Volume | 14 fichiers (p00 : 1 enrôlement + 8 tests ; p01 : 1 enrôlement + 4 tests) |
| Format | wav 16 kHz mono, enregistré avec le micro intégré du PC |
| Minimisation | Codes p00 / p01 ... au lieu des noms ; genre non précisé ; fichiers locaux, jamais publiés, supprimés à la fin du projet ou sur demande |

**Limite :** Compliqué de développer cela à une plus grande échelle et 2 personnes ne suffisent pas pour faire des statistiques. Ce corpus sert surtout d'illustration qualitative et à la carte t-SNE.

## 4. Corpus envisagés et non retenus

| Corpus | Raison |
|---|---|
| **VoxCeleb** (téléchargement direct comme pour Common Voice 26.0) | Inutile car ECAPA a déjà été entraîné dessus. Il est utilisé à travers le modèle pré-entraîné. |
| **Common Voice dans d'autres langues** (ex : EN/ES) | Chaque locuteur Common Voice ne parle qu'une langue. Entraîner une couche en multilingue lui apprendrait indirectement le raccourci « une langue différente = une personne différente ». |

## 5. Risques de fuite et parades

| Risque | Parade |
|---|---|
| Un même locuteur dans deux groupes | Découpage par locuteur et test automatique d'intersection vide |
| Locuteurs cv26 de validation ou de test présents dans cv27 | Exclusion par `client_id` et par noms de clips, vérification par groupe sur le PC |
| Réglages faits sur les données de test | Tous les choix sont faits sur la validation ; le test reste fermé jusqu'au test final |
| Un clip de test qui sert aussi à l'enrôlement (banc v3) | Pour les paires positives, le clip de test est exclu du bloc d'enrôlement |
| Une personne de la grande validation cv27 déjà utilisée pour calculer la moyenne soustraite (train cv26) | Ces personnes sont placées d'office dans le groupe d'entraînement cv27, jamais en validation |

## 6. Limites générales des données

- **Uniquement du français** lu pour toutes les mesures chiffrées. Lors de la démo, la personne est libre d'utiliser de la parole spontanée, dans plusieurs langues.
- **Pas de voix d'enfants** identifiées : l'âge déclaré le plus jeune dans Common Voice est « adolescent ». La reconnaissance des voix d'enfants n'est donc pas mesurée et ECAPA a lui aussi été entraîné sur des adultes.
- **Micro et pièce inconnus** dans Common Voice.
- **Taille du banc de validation cv26** : 320 paires positives et 320 voix proches, soit une incertitude d'environ ±1 point sur les taux d'erreur. Les seuils ont donc été vérifiés sur la grande validation cv27 (290 personnes, près de 84 000 essais d'imposteurs proches).
- **Taille du test final** : 198 essais de la vraie personne seulement, soit une incertitude d'environ ±4.5 points sur le taux de vraies personnes refusées.
- **Bruit simulé** et non enregistré en conditions réelles.
