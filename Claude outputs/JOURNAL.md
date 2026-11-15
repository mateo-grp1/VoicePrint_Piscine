# Journal de bord : tout ce qui a été fait, étape par étape

Ce journal raconte le projet dans l'ordre chronologique, **au niveau du code** : pour chaque étape, ce que fait chaque script, avec quels réglages, ce qu'on a obtenu et pourquoi on a pris chaque décision. Les chiffres de synthèse sont dans [`RAPPORT.md`](RAPPORT.md).

Conventions utilisées partout :
- Tous les scripts se lancent depuis la racine (`python src/<script>.py`).
- Graine aléatoire **42** partout, pour des résultats reproductibles.
- Une seule fonction d'évaluation : `src/evaluate.py`.
- Chaque résultat est ajouté à `results/resultats.csv` avec `enregistrer()`.

---

## Étape 0 : mise en place du projet

*Commits « Structure initiale du projet », « Ajout des dépendances ».*

- **Environnement** : Windows 11, VS Code, PowerShell, Python 3.11.9 dans un environnement virtuel `.venv`, sans GPU. Noyau Jupyter « Voiceprint (.venv) ».
- **Arborescence** : `src/` pour les scripts, `notebooks/` pour l'exploration, `data/` (hors GitHub), `models/` (hors GitHub), `results/` (versionné), `docs/`.
- **`.gitignore`** : `.venv/`, `__pycache__/`, `data/`, `models/`, `*.wav`, `*.mp3`, `*.flac`, `*.m4a`, `*.npy`. Aucun audio ni aucune empreinte ne peut partir sur GitHub.
- **Dépendances** figées dans `requirements.txt` : torch 2.15.0, torchaudio 2.11.1, speechbrain 1.1.2, librosa 0.11.0, scikit-learn 1.9.1, pandas, matplotlib, streamlit 1.67.0, praat-parselmouth 0.4.7, sounddevice 0.5.6.

---

## Étape 1 : choix et préparation des données (Common Voice 26.0)

*Commits « Script de création de metadata.csv », « Conversion wav 16 kHz et séparation par locuteur (60/20/20, seed 42) ».*

**Choix du corpus.** VoxCeleb et Common Voice ont été examinés, comme le demande le sujet.
- Common Voice français de France 26.0 est retenu : licence CC0, téléchargement simple, et un identifiant de locuteur (`client_id`) qui permet de constituer des paires.
- VoxCeleb sera utilisé **indirectement**, puisque c'est le corpus d'entraînement d'ECAPA.

**Exploration** (`notebooks/01_inspecter_cv26.ipynb`) : lecture des `.tsv`, nombre de clips par locuteur, durées, métadonnées âge et genre (souvent vides). Constat clé : les contributions sont très inégales, de 5 à plus de 500 clips par personne.

**`src/make_metadata.py`**
- Lit `train.tsv`, `dev.tsv` et `test.tsv` (`pandas.read_csv(sep="\t", quoting=3)`, car les phrases contiennent des guillemets) et les concatène. Le split officiel est gardé dans une colonne `split_officiel`, mais il ne sera pas utilisé.
- Garde les locuteurs qui ont **au moins 5 clips**.
- Remplace le `client_id` (long hachage) par un code court `cv_000`, `cv_001`… dans un ordre trié, donc stable.
- Écrit `data/metadata.csv` (`file_path`, `speaker_id`, `corpus`, `language`, `gender`, `age`, `split_officiel`) et vérifie que tous les fichiers existent.

**`src/convert_audio.py`**
- Convertit chaque MP3 en **wav mono 16 kHz PCM 16 bits** avec `ffmpeg` (`-ac 1 -ar 16000 -c:a pcm_s16le`). On peut relancer le script : les fichiers déjà convertis sont sautés.
- Lit la durée dans l'en-tête (`soundfile.info`), retire les clips de **moins de 2 s**, puis les locuteurs qui n'ont plus 5 clips.
- Résultat : **5 282 clips, 142 locuteurs, 7,45 h**, avec une durée moyenne de 5,1 s.

**`src/split_speakers.py`** : on découpe les **locuteurs**, pas les clips.
- La liste des locuteurs est triée, mélangée avec `np.random.default_rng(42)`, puis coupée en 60/20/20.
- Chaque clip reçoit le groupe de son locuteur (colonne `split`).
- Trois `assert` vérifient l'absence de fuite :
  1. aucune intersection entre les groupes ;
  2. tous les clips ont un groupe ;
  3. aucun locuteur n'a des clips dans deux groupes.
- Résultat : train **85** locuteurs (3 361 clips), validation **28** (601), test **29** (1 320).

**`src/make_pairs.py`**
- **Positives** : pour chaque locuteur, toutes les combinaisons de 2 clips sont énumérées (`itertools.combinations`), puis on en tire au plus 50 (train) ou 20 (validation et test). Ce plafond limite le poids des gros contributeurs.
- **Négatives** : on tire 2 locuteurs différents, puis un clip de chacun, sans doublon. Il y a autant de négatives que de positives.
- `assert` sur les groupes, les étiquettes et l'absence de clip comparé à lui-même.
- Résultat : `pairs_train.csv` 5 394, `pairs_val.csv` 710, `pairs_test.csv` 940.

---

## Étape 2 : la fonction d'évaluation unique

**`src/evaluate.py`**, utilisé par **tous** les scripts :
- `evaluate(scores, labels, nom=None)` :
  - calcule la courbe ROC (`sklearn.metrics.roc_curve`) ;
  - **EER** : le point où |faux rejets − fausses acceptations| est minimal ;
  - **minDCF** : coût = 1 × FNR × 0,05 + 1 × FPR × 0,95, normalisé par min(0,05 ; 0,95), et pris au meilleur seuil ;
  - **AUC** ;
  - **`frr_far1`** : les faux rejets au seuil qui n'accepte au plus que 1 % d'imposteurs, avec ce **seuil** (`seuil_far1`). Ajouté au jour 3 ;
  - trace la ROC dans `results/roc_<nom>.png` si `nom` est donné.
- `enregistrer(nom, split, res, notes)` ajoute une ligne à `results/resultats.csv`. Il accepte de nouvelles colonnes sans décaler le tableau.

---

## Étape 3 : baseline explicable (MFCC)

*Commits « Test et résultat 1ère ablation (normalisation) », « Classifieurs simples (LogReg, SVM) sur paires MFCC », « petit réseau mlp sur paires mfcc (fin baseline de base) ».*

**Exploration** (`notebooks/02_baseline_explorer.ipynb`) : spectrogrammes, MFCC et statistiques acoustiques de quelques clips, pour choisir les réglages.

**`src/features_mfcc.py`**
- Pour chaque clip : `librosa.feature.mfcc(n_mfcc=20, n_fft=512, hop_length=160)`, soit des fenêtres de 32 ms tous les 10 ms à 16 kHz.
- Vecteur du clip = **moyenne et écart-type** de chaque coefficient dans le temps, soit **40 nombres**.
- Sauvegarde dans `data/features/mfcc40.npz` (`X`, `paths`).

**`src/baseline_cosine.py`** : première ablation.
- Version 1 : cosinus entre les vecteurs bruts → **EER 19,0 %**.
- Version 2 : standardisation (moyenne et écart-type calculés **sur le train uniquement**), puis cosinus → **EER 11,0 %**.
- Explication : le MFCC 0 représente l'énergie, donc le volume. Ses valeurs sont énormes par rapport aux autres et il domine le cosinus. Standardiser met tous les coefficients à la même échelle.

**`src/pair_features.py`** (outils partagés)
- `charger_mfcc_normalises()` : les MFCC standardisés avec les statistiques du train.
- `entrees_paires(..., mode)` : transforme une paire (a, b) en entrée de classifieur, soit `|a − b|` (40 nombres), soit `|a − b|` suivi de `a × b` (80 nombres). Ces deux formes sont symétriques : l'ordre des deux clips ne change rien.

**`src/baseline_classifiers.py`**
- `make_pipeline(StandardScaler(), modèle)`, entraîné sur `pairs_train`, évalué sur `pairs_val` avec `decision_function` comme score.
- Résultats : régression logistique |a−b| 12,3 %, régression logistique |a−b| + a×b 8,0 %, **SVM RBF (C = 1) 6,8 %**.

**`src/baseline_mlp.py`** (petit réseau PyTorch, demandé par le sujet)
- Mêmes entrées de 80 nombres, de nouveau standardisées sur le train.
- Réseau : `Linear(80, 64) → ReLU → Dropout(0,3) → Linear(64, 32) → ReLU → Linear(32, 1)`.
- `BCEWithLogitsLoss`, Adam (lr = 1e-3, weight_decay = 1e-4), 60 epochs, lots de 64, `torch.manual_seed(42)`.
- À chaque epoch : loss train, loss validation et EER validation. On garde les poids de la **meilleure loss de validation** (`copy.deepcopy(state_dict)`).
- Figure `results/mlp_courbes.png` : loss train et validation, et EER par epoch. Poids sauvegardés dans `models/mlp_mfcc.pt`.
- Résultat : **EER 7,2 %**, minDCF 0,332, AUC 0,976.

---

## Étape 4 : empreintes pré-entraînées ECAPA-TDNN et démo v0

*Commit « Essai des evaluation avec le modele ECAPA (Démo v0) ».*

**Exploration** (`notebooks/03_test_ecapa.ipynb`) : chargement du modèle, empreinte d'un clip, premiers scores.

**`src/ecapa_eval.py`**
- `charger_modele()` : `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="models/ecapa")`.
- `calculer_empreintes(chemins, model)` : pour chaque fichier, `librosa.load(sr=16000)` puis `model.encode_batch(...)`, soit 192 nombres. Toutes les empreintes sont gardées dans un **cache** `data/embeddings/ecapa.npz` (`paths`, `emb`) : seules les nouvelles sont calculées.
- `scores_cosinus(paires, emb)` : normalise chaque empreinte à une longueur de 1, puis fait le produit scalaire.
- Évaluation sur **exactement** `pairs_val.csv` → **EER 2,1 %**, minDCF 0,169, AUC 0,996. Score moyen même personne 0,688 ; personnes différentes 0,147.

**Démo v0** (Streamlit) : enrôlement de 20 s, 10 essais réels de l'auteur, tous corrects, y compris en espagnol, en anglais, en fausse langue et loin du micro.

**Décision** : approche principale = **ECAPA gelé + adaptations légères**. La baseline sert de comparaison et de base pour la partie explicable, mais elle n'apparaît pas dans l'interface de la démo.

---

## Étape 5 : un banc d'essai plus exigeant (banc v2)

*Commit « III) 1ere etape banc d'essai ».*

**Pourquoi** : 2,1 % d'EER sur des imposteurs tirés au hasard, c'est trop facile. Le jour J, il y aura des voix proches et des enregistrements courts.

**`src/build_bench_v2.py`**
1. Reprend les locuteurs et les clips de `pairs_val.csv` (validation uniquement).
2. Mesure la **hauteur médiane** de chaque locuteur sur 5 clips au plus, avec `librosa.pyin(fmin=60, fmax=400)` sur les seules trames voisées. Écrit `data/features/f0_locuteurs_val.csv`.
3. Positives : jusqu'à 15 combinaisons de 2 clips par locuteur, soit 320.
4. Négatives **au hasard** : 320. Négatives **difficiles** : les couples de même genre (quand il est connu) sont triés par écart de hauteur, on garde le **quart le plus proche** (écart de 0 à 14 Hz), puis on tire 320 paires.
5. Pour chaque paire, 3 versions : **entier**, puis extraits de **3 s** et **2 s** pris **au milieu** du clip, écrits dans `data/processed/extraits_3s|2s/`.
6. Sortie : `data/pairs/pairs_val_v2.csv` (colonnes `file1`, `file2`, `spk1`, `spk2`, `label`, `type`, `duree`).

**`src/escapa_bench.py`** (nom de fichier avec une coquille, le contenu est `ecapa_bench`) : ECAPA brut par durée et par type de négatives. Voix proches : EER 4,2 % (entier comme 3 s), minDCF 0,225 (entier) à 0,272 (3 s).

---

## Étape 6 : réglages d'ECAPA et durée d'enrôlement

*Commit « III) 2eme etape choix technique : brut + soustraction moyenne avec enrollement 3 ou 8 s ».*

**`src/normalisation.py`** (outils partagés, utilisés ensuite par la démo)
- `clips_train()` : 10 clips par locuteur de **train**, avec un mélange fixe.
- `unitaire(E)` : chaque empreinte ramenée à une longueur de 1.
- `preparer(E, moyenne)` : longueur 1 → soustraction de l'empreinte moyenne du train → longueur 1.
- `construire_reference(emb, train)` : l'empreinte moyenne du train et une **cohorte** (une empreinte moyenne par locuteur de train), qui sert pour l'AS-norm.
- `as_norm(scores, A, B, cohorte, top_k=30)` : chaque score est comparé aux 30 imposteurs de la cohorte les plus proches de A et de B, puis on fait la moyenne des deux scores centrés-réduits.
- `seuil_validation(duree)` : relit le seuil à 1 % d'imposteurs dans `results/enrolement_v2.csv`.

**`src/reglages_v2.py`** : on change un seul réglage à la fois (brut, + moyenne, + AS-norm, + les deux), sur toutes les durées.
- Voix proches, 3 s : brut minDCF 0,272 → **moyenne 0,172** ; AS-norm 0,337.
- Le refus de vraies personnes à 1 % d'imposteurs passe de 12,2 % à **8,1 %** avec la moyenne, mais monte à 27,5 % avec l'AS-norm.
- **Décision** : moyenne gardée, AS-norm écartée, car elle laisse passer des imposteurs à score élevé.

**`src/enrolement_v2.py`** : la durée d'enrôlement, dans les conditions de la démo.
- Pour chaque locuteur de validation, ses clips sont mis bout à bout (silences du début et de la fin retirés avec `librosa.effects.trim(top_db=30)`) jusqu'à obtenir au moins 10 s. Ce bloc sert à l'enrôlement ; les clips restants servent au test.
- Enrôlement de **3, 5, 8 et 10 s** (on coupe le bloc), selon deux calculs :
  - « bloc » : une seule empreinte ;
  - « morceaux » : moyenne des empreintes de morceaux de 3 s décalés de 1,5 s.
- Tests : extraits de 3 s et clips entiers. Positifs : ses autres clips ; imposteurs : tous les clips des autres locuteurs. Voix proches selon la même règle que le banc v2.
- Résultats (bloc, moyenne soustraite, test 3 s, voix proches) : 3 s → EER 2,6 %, FRR@FA1% 3,6 %, seuil 0,362 ; 10 s → EER 1,0 %, FRR 0 %, seuil 0,378.
- **Décisions** : enrôlement d'**un seul bloc** (bloc et morceaux sont équivalents), de **3 s** ; 2 s est abandonné.
- Une première version de ce script, qui enrôlait 1, 2, 4 ou 6 clips entiers, reste visible dans `resultats.csv` (lignes « enrôlement N clips »). Elle a été remplacée par la version en secondes, plus proche de la démo.

---

## Étape 7 : mini-corpus et partie explicable

**`src/enregistrer_corpus.py`** (`sounddevice` et `soundfile`)
- Affiche le texte de consentement et demande « oui » ; sans accord, le script s'arrête.
- Demande un code (`p00`, `p01`…), jamais un nom.
- Protocole guidé avec compte à rebours, réécoute et possibilité de recommencer :
  - enrôlement de 10 s ;
  - tests de 5 s : 2 par langue (jusqu'à 3 langues), 1 en fausse langue, 1 à environ 1,5 m du micro.
- Écrit des wav 16 kHz mono dans `data/mini_corpus/<code>/` et une ligne par fichier dans `data/mini_corpus/metadata.csv`.
- Résultat : `p00` (fr, es, en) et `p01` (fr), 12 prises de test au total.

**`src/eval_mini_corpus.py`** : vérification, sans aucun réglage.
- Enrôlement = 3 premières secondes de parole ; test = 3 s prises au milieu de la parole ; moyenne soustraite ; seuil de validation de 0,362.
- Chaque personne est comparée à ses propres prises, aux prises de l'autre personne, et à 564 extraits Common Voice de validation.
- Résultat : 12/12 vraies prises acceptées (scores de 0,382 à 0,627), 0/12 proches acceptés (maximum 0,088), 2/564 imposteurs Common Voice acceptés.

**`src/caracteristiques.py`** : des caractéristiques qu'on peut **nommer**.
- Avec Praat via `parselmouth` :
  - hauteur (`to_pitch`, 60-400 Hz) : médiane et variation en demi-tons ;
  - formants F1, F2 et F3 (`to_formant_burg`, 5 formants, plafond 5 500 Hz), mesurés aux instants voisés ;
  - jitter et shimmer (`To PointProcess (periodic, cc)`) ;
  - HNR (`To Harmonicity (cc)`).
- Avec librosa : centroïde spectral et MFCC 1 à 12 (le MFCC 0, qui mesure le volume, est volontairement exclu), sur les trames actives, et rythme (nombre d'attaques par seconde).
- `expliquer(a, b, ecart_intra)` renvoie les 3 caractéristiques dont l'écart, divisé par l'écart habituel chez une même personne, est le plus grand. C'est le panneau « Pourquoi ? » de la démo.

**`src/explicable.py`**
1. Mesure des caractéristiques sur 10 clips par locuteur de train et sur tous les clips de validation (cache `data/features/caracteristiques.csv`).
2. **Ratio de Fisher** sur le train = variance des moyennes par locuteur / moyenne des variances par locuteur. Résultats : F0 10,6 ; MFCC 1,6 à 3,9 ; HNR 2,4 ; … ; F2 0,69 ; rythme 0,51. Calcul identique sur les 192 dimensions d'ECAPA : moyenne 1,76, maximum 3,04.
3. `results/ecart_intra_locuteur.csv` : l'écart habituel chez une même personne, utilisé par la démo.
4. **Plus proche voisin** sur 282 clips de validation : caractéristiques nommées 63 %, avec les MFCC 93 %, ECAPA **98 %**.
5. **t-SNE** (perplexité 30, initialisation PCA) des empreintes de validation et du mini-corpus : `results/tsne_ecapa.png`. Les 9 prises de `p00` dans 3 langues forment un seul groupe.

---

## Étape 8 : robustesse au bruit, calibration et démo v1

*Commit « V) Test robustesse au bruit (--) ajout démo v1 ».*

**`src/augmentation.py`** (outils partagés)
- `bruit_rose(n, rng)` : bruit blanc dont le spectre est divisé par √f, ce qui donne plus d'énergie dans les graves (ventilation, salle).
- `brouhaha(n, voix, rng, nb=4)` : 4 voix de **train** superposées, chacune normalisée par son écart-type.
- `melanger(y, b, snr_db)` : ajoute le bruit au rapport signal/bruit voulu, en mettant le bruit à l'échelle par √(P_signal / (P_bruit × 10^(snr/10))).
- `empreintes_signaux(model, signaux, lot=32)` : empreintes par lots, avec remplissage et longueurs relatives (`wav_lens`).

**`src/bruit_v2.py`** : enrôlement 3 s propre, test 3 s bruité (ambiance ou brouhaha à 20, 10 et 5 dB), voix proches, seuil de la démo.
- Le bruit fait baisser **tous** les scores : moins d'imposteurs acceptés, mais beaucoup plus de vraies personnes refusées.
- Brouhaha à 10 dB : 23,1 % de vraies personnes refusées ; à 5 dB : 41,6 %.

**`src/calibration.py`** (remplacé ensuite par `eval_v3.py`)
- Régression logistique `class_weight="balanced"` qui transforme le score en probabilité, apprise sur les paires de validation de 3 s.
- Vérification honnête : apprise sur la moitié des locuteurs, testée sur l'autre moitié.
- Écrit `models/reference_demo.npz` : moyenne, seuil, `calib_a = 10,96`, `calib_b = −3,26`, écarts habituels.

**`app.py` (démo v1, Streamlit)**
- `lire_audio()` : lecture du micro en 16 kHz mono, puis retrait de **tous** les silences et pauses (`librosa.effects.split(top_db=30)`). Les morceaux de parole sont recollés.
- Enrôlement :
  - choix de la durée parmi 3, 5 et 10 s ;
  - case de consentement obligatoire ;
  - message d'erreur s'il n'y a pas assez de parole ;
  - on garde les N premières secondes de parole ;
  - l'empreinte et les caractéristiques nommées sont stockées dans `data/enrolement/empreinte.npz`.
- Vérification :
  - test jusqu'à 10 s de parole ;
  - score cosinus après `preparer()` ;
  - décision avec le seuil correspondant à la durée d'enrôlement ;
  - confiance affichée seulement si la personne est acceptée ;
  - panneau « Pourquoi ? » ;
  - **passages enchaînés** avec un historique (`st.session_state`).
- Suppression : un bouton qui efface le fichier d'empreinte.
- `lire_npz()` copie le contenu puis **ferme** le fichier, sinon Windows interdit de le supprimer.
- Conseils de qualité (saturation, rapport signal/bruit) codés mais **désactivés** (`AFFICHER_CONSEILS = False`) : le micro intégré du PC déclenchait de fausses alertes de saturation.

Tests de l'auteur :
- enrôlement de 5 ou 10 s : tout passe ;
- enrôlement de 3 s : quelques refus quand l'enrôlement est en anglais ou en espagnol et le test en français. Le retrait des pauses internes (et pas seulement des silences de début et de fin) a amélioré ces cas.

---

## Étape 9 : metric learning v1 (triplet loss) et diagnostic d'overfit

**`src/metric_learning.py`**
- **Données d'entraînement** : 85 locuteurs de train cv26, au plus 20 clips chacun, soit 1 005 clips.
- Pour chaque clip, 6 versions (2 propres, ambiance 20 et 10 dB, brouhaha 20 et 10 dB). Chacune utilise un **extrait de 3 s au hasard** (`librosa.effects.trim`), et le brouhaha est fait de 60 voix de train.
- Empreintes ECAPA par lots, en cache (`data/embeddings/ml_train.npz`), soit 6 030 empreintes. Moyenne soustraite.
- **Validation** : paires de 3 s du banc v2, enrôlement propre, test propre, ambiance à 10 dB ou brouhaha à 10 dB (cache `ml_val_bruit.npz`).
- **Couche** : `F.normalize(x + Linear(192, 192)(x))`, avec des poids **initialisés à zéro**, donc sortie = entrée au départ.
- **Loss** : triplet « batch-hard », marge 0,2. Distance = 1 − cosinus. Pour chaque ancre, on prend le positif le plus éloigné et le négatif le plus proche du lot.
- Lots de 16 personnes × 6 empreintes, Adam (lr = 1e-3, weight_decay = 1e-3), 40 epochs.
- Epoch gardée selon la moyenne des minDCF propre et brouhaha.
- Résultat : la loss passe de 0,07 (epoch 5) à 0,027 pendant que la validation se dégrade (minDCF propre 0,172 → 0,244). C'est de l'**overfit**.
- Causes identifiées :
  1. 85 personnes seulement ;
  2. une loss trop facile ;
  3. une couche trop libre (environ 37 000 paramètres) ;
  4. un critère de choix instable.

---

## Étape 10 : nouveau corpus pour le metric learning v2 (Common Voice 27.0 sur Colab)

*Commits « commit du nouveau corpus (800 locuteurs de Common Voice 27.0) » et « Essaye de la v3… ».*

**Recherche** sur Mozilla Data Collective :
- *Scripted Speech 27.0 French* (28,55 Go, 21 116 locuteurs) est retenu ;
- *Spontaneous Speech 5.0 French* (52 locuteurs) est envisagé pour un test, puis abandonné.

**`src/exporter_exclusions.py`** (sur le PC, avant Colab)
- Relit `metadata.csv` et prend tous les clips **hors train**, c'est-à-dire la validation et le test.
- Retrouve leur `client_id` dans les `.tsv` bruts de cv26, car `speaker_id` est un code court.
- Exporte `data/exclusions_cv26.csv` avec deux types d'identifiants : les `client_id` et les noms de clips (`common_voice_fr_XXXX`), qui restent les mêmes d'une version de Common Voice à l'autre.
- Le fichier est déposé dans `Mon Drive/voiceprint/`.

**`notebooks/04_preparer_cv27_colab.ipynb`** (Colab, GPU T4), 12 cellules :
1. Réglages : 1 000 locuteurs visés, 3 000 candidats, 15 clips extraits par candidat, 12 gardés au plus, 8 au minimum, seuil grave/aigu à 165 Hz, extraits de 3 s, graine 42.
2. Montage de Google Drive et vérification du fichier d'exclusions.
3. Installation de `speechbrain==1.1.2` et de `praat-parselmouth`.
4. Fonction `telecharger()` :
   - `POST /api/datasets/<id>/download` avec la clé lue dans les **Secrets** de Colab (jamais écrite dans le code) ;
   - téléchargement en continu depuis l'URL signée renvoyée ;
   - **reprise** après coupure avec l'en-tête `Range` ;
   - messages clairs pour les erreurs 401, 403 et 429.
   - *Problème rencontré* : l'ancienne adresse de l'API redirigeait, et la redirection transformait le POST en GET, d'où une erreur 405. *Correction* : nouvelle adresse `mozilladatacollective.com/api` et `allow_redirects=False`.
5. Téléchargement de l'archive. *Problème rencontré* : coupure à 25,4 Go sur 30,7 Go ; la cellule relancée a repris là où elle s'était arrêtée.
6. **Passage 1** : `tar -xzf --wildcards "*.tsv"` n'extrait que les tableaux, pas les 870 000 audios.
7. Sélection :
   - clips validés de 3,8 s ou plus (`clip_durations.tsv`) ;
   - **anti-fuite** par `client_id` **et** par noms de clips : 57/57 et 1 921/1 921 retrouvés, 57 locuteurs exclus ;
   - le script s'arrête si rien n'est retrouvé, car on ne pourrait alors rien garantir ;
   - une phrase différente par clip ;
   - 4 912 locuteurs éligibles, dont 3 000 tirés au hasard ;
   - `assert` d'intersection vide avec les exclus.
8. **Passage 2** : `tar -xzf -T liste_clips.txt` n'extrait que les 44 016 clips choisis.
9. **Hauteur** de chaque candidat :
   - préparation : `librosa.load(16 kHz)`, `effects.split(top_db=30)`, 8 s au plus ;
   - Praat `to_pitch(60-500 Hz)`, médiane sur 2 clips ;
   - calcul parallèle (`multiprocessing.Pool`) ;
   - résultat : 2 169 voix graves et 831 aiguës, puis **équilibrage à 500 + 500**.
10. Préparation de la parole des 1 000 locuteurs gardés : au moins 3 s de parole par clip, au plus 12 clips, au moins 8 clips par locuteur, soit **945 locuteurs** et 10 353 clips.
    - `selection_cv27.csv` ne contient que des colonnes minimales (minimisation).
    - L'audio est rangé dans **un seul zip** de 451 Mo sur le Drive (écrire des milliers de petits fichiers sur Drive est très lent).
11. **Empreintes sur GPU** :
    - 6 versions par clip, chacune sur un extrait de 3 s au hasard ;
    - bruit rose obtenu par filtrage en 1/f dans le domaine fréquentiel ; brouhaha de 5 voix d'**autres** locuteurs de l'entraînement ;
    - lots de 240, en 402 s, pour 62 118 × 192 empreintes ;
    - `emb_cv27_train.npz` contient `emb`, `locuteur`, `clip`, `version` et `noms_versions` ;
    - un **signal de contrôle** synthétique (`default_rng(0)`) est encodé et sauvé dans `controle_ecapa.npy`.
12. Bilan et **suppression de l'archive de 28 Go**.

*Problème rencontré* : la bibliothèque `speechbrain` importée avant le calcul parallèle peut casser l'envoi des fonctions aux processus (un module chargé « paresseusement »). Elle n'est donc importée qu'à la cellule 11.

**`src/verifier_cv27.py`** (sur le PC)
- Compare la sélection à **chaque groupe** cv26 : 0 locuteur commun avec la validation et avec le test, 5 avec le train (autorisé). Écrit `data/cv27/locuteurs_a_retirer.csv`, vide.
- Recalcule l'empreinte du signal de contrôle sur le CPU : **cosinus Colab / PC = 1,00000**.

---

## Étape 11 : metric learning v2 (AAM-softmax, 945 locuteurs)

**`src/metric_learning_v2.py`**
- Charge les empreintes cv27 (`allow_pickle=True` pour les noms de clips, sans risque puisque c'est notre propre fichier) et leur soustrait la **même moyenne que la démo**.
- **10 % des locuteurs (94) sont tenus à l'écart** pour la surveillance. Avec eux, on construit 3 000 paires par condition : enrôlement propre contre test propre, ambiance ou brouhaha, toujours sur deux clips différents.
- Validation : le **même banc** que la v1, avec les mêmes caches, pour une comparaison directe.
- Deux couches, toutes deux à « ne rien changer » au départ :
  - `Diagonale` : `normalize(x × w)` avec w initialisé à 1 ; pénalité Σ(w − 1)² ;
  - `ResiduelleRang` : `normalize(x + haut(bas(x)))`, avec 192 → 32 → 192 et `haut` initialisé à zéro ; pénalité Σ(haut)².
- **Loss AAM-softmax** :
  - chaque personne d'entraînement a un « centre », initialisé à son empreinte moyenne ;
  - logit de la bonne classe = 30 × cos(θ + 0,2), les autres = 30 × cos θ ;
  - entropie croisée + 0,01 × pénalité.
- Adam : lr 1e-3 pour la couche, 1e-2 pour les centres. Lots de 512, 30 epochs.
- Choix de l'epoch : minDCF moyen sur toutes les paires de 3 s (3 conditions), **lissé** sur 3 epochs.
- Verdict automatique : la couche est gardée seulement si elle améliore à la fois le minDCF moyen et le FRR@FA1% moyen, sans dégrader le cas propre de plus de 0,01.
- *Problème rencontré* : `ValueError: Object arrays cannot be loaded when allow_pickle=False`, car les noms de clips ont été enregistrés comme objets Python sur Colab. Corrigé.
- Résultats :
  - diagonale : minDCF brouhaha 0,266 → 0,228, mais FRR propre 8,1 % → 9,1 % et ambiance 12,8 % → 14,7 % ;
  - résiduelle : epoch gardée 0 ;
  - **aucune n'est retenue**.
- Constat clé : sur les 94 personnes tenues à l'écart, la couche résiduelle **progresse** (EER 3,3 % → 2,4 %). L'overfit sur les personnes est corrigé, mais le gain ne se transfère pas au banc cv26.

---

## Étape 12 : une seule recette de préparation (banc v3) et réglages finaux

**Constat** : `build_bench_v2.py` prend le **milieu du clip brut**, silences compris. La démo retire tous les silences, puis prend les N premières secondes **de parole**. Les seuils étaient donc mesurés dans des conditions différentes de celles du jour J.

**`src/preparation.py`** : la recette unique, importée par la démo **et** par l'évaluation.
- `charger_brut(source)` : 16 kHz mono.
- `garder_parole(brut)` : `librosa.effects.split(top_db=30)`, morceaux recollés.
- `enrolement(parole, duree)` : les N premières secondes.
- `test(parole, 10)` : jusqu'à 10 s.
- `empreinte_brute(modele, y)` : un seul appel ECAPA, sans remplissage, pour que le résultat soit identique dans la démo et dans le banc.

**`src/build_bench_v3.py`**
- Reprend les **mêmes 960 paires** (version « entier » du banc v2).
- Colonne `exclu` : pour une paire positive, le clip de test n'entre jamais dans l'enrôlement.
- Enrôlement : la parole du clip `file1`, puis si besoin celle des autres clips du même locuteur, coupée à 3, 5 ou 10 s.
- Test : le clip `file2` dans 3 conditions (propre, ambiance 10 dB, brouhaha 10 dB). Le bruit est ajouté au **signal brut, avant `garder_parole`**, comme dans une vraie salle. Puis le test est coupé à 3 s ou jusqu'à 10 s.
- Le script calcule aussi les clips de train cv26 préparés comme la démo, pour une moyenne « même recette ».
- Cache par clé (`enrol|3|file1|exclu`, `test|max10s|brouhaha|file2`, `train|chemin`) dans `data/embeddings/banc_v3.npz`, sauvegardé toutes les 250 empreintes.
- 3 588 empreintes, en environ 40 min sur le CPU. 21 enrôlements sont un peu plus courts que prévu, faute de parole.

**`src/eval_v3.py`**
1. Compare **trois moyennes** : cv26 brute, cv26 préparée comme la démo, cv27. La meilleure est **cv26 préparée démo** (minDCF moyen 0,217 ; FRR 8,2 %).
2. Re-teste les couches v2 sur le banc v3 : aucun gain fiable.
3. Produit le tableau complet (enrôlement × test × condition) dans `results/banc_v3.csv` et les lignes de `resultats.csv`.
4. Calcule les **seuils** (1 % d'imposteurs, voix proches, le plus prudent des tests de 3 s et jusqu'à 10 s) : **0,416 / 0,447 / 0,482**.
5. Fait la **calibration** (enrôlement 3 s, test jusqu'à 10 s, toutes les paires), vérifiée sur une moitié de locuteurs non vus : a = 10,54, b = −3,38, soit 73 % au seuil.
6. Écrit `models/reference_demo.npz` (moyenne, seuils, calibration, écarts habituels, `banc="v3"`, `moyenne_source`). L'ancien fichier est gardé sous `reference_demo_v1.npz`.

**`app.py`** est mis à jour :
- il importe `preparation.py` au lieu de sa propre copie du nettoyage ;
- il lit les seuils dans `reference_demo.npz` (repli sur l'ancien calcul si le fichier est ancien) ;
- il affiche « banc v3 » à côté des seuils.

Résultat final (validation, enrôlement 3 s, voix proches, test jusqu'à 10 s, propre) : **EER 1,9 %, minDCF 0,172, 5,3 % de vraies personnes refusées** pour 1 % d'imposteurs acceptés.

---

## Étape 13 : à venir

[À COMPLÉTER au fil des derniers jours]
- Plan de test de la démo avec les seuils du banc v3 : enrôlement en français puis en anglais, passages en 3 langues, loin du micro, autres personnes.
- Analyse des cas d'échec : écouter les pires paires de validation.
- **Test final** sur `pairs_test.csv` (29 locuteurs jamais utilisés), avec la configuration figée, une seule fois.
- Bonus : test de rejeu (un enregistrement joué sur un téléphone).
- Répétition de la démo avec plusieurs personnes.
