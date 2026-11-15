# Voiceprint : rapport technique

*Piscine IA Lab. Vérification du locuteur sur des personnes inconnues.*

## Résumé

On construit un système qui décide si deux enregistrements viennent de la même personne, et qui doit fonctionner sur des **personnes jamais entendues**. On compare une baseline classique (MFCC, cosinus, classifieurs, petit réseau) à des empreintes **ECAPA-TDNN** pré-entraînées, sur les **mêmes paires** et avec la **même fonction d'évaluation**.

ECAPA divise l'erreur par plus de trois (EER de 2,1 % contre 6,8 % pour la meilleure baseline). Trois réglages sont mesurés par ablation : la normalisation, la durée d'enrôlement et la robustesse au bruit. Deux versions de metric learning sont essayées : la première sur-apprend, la seconde corrige l'overfit mais n'apporte pas de gain fiable. Elles ne sont donc pas retenues.

Enfin, un banc d'essai **préparé exactement comme la démo** (banc v3) fixe les réglages finaux : enrôlement de **3 s de parole**, seuil de **0,416**, **EER de 1,9 %** et **5,3 % de vraies personnes refusées** pour 1 % d'imposteurs acceptés parmi les voix proches (validation).

---

## 1. Problème et protocole

**Tâche.** Vérification (et non identification) : une personne s'enrôle avec un seul enregistrement, puis chaque nouvel enregistrement est comparé à cette empreinte. Le système répond « même personne » ou « autre personne ».

**Contrainte de la démo.** Une personne inconnue s'enrôle en **3 s** environ (5 s en repli, jamais plus de 10 s). Ensuite, 5 à 10 personnes variées (âges, genres, langues) parlent chacune environ 10 s. Le système doit reconnaître la personne enrôlée et ne pas réagir aux autres. **L'erreur la plus grave est d'accepter un imposteur.**

**Données.** Common Voice 26.0, français de France : 142 locuteurs, découpés **par locuteur** en 85 (train), 28 (validation) et 29 (test). Voir [`FICHE_CORPUS.md`](FICHE_CORPUS.md).

**Mesures** (une seule fonction, `src/evaluate.py`)
- **EER** (*Equal Error Rate*) : le taux d'erreur au seuil où les fausses acceptations égalent les faux rejets. Plus il est bas, mieux c'est.
- **minDCF** (p_cible = 0,05, coûts égaux) : le coût minimal, de 0 (parfait) à 1 (aussi mauvais que tout refuser). Il pénalise surtout les **imposteurs acceptés**.
- **AUC** et **courbe ROC**.
- **FRR@FA1%** : la part de vraies personnes refusées quand on n'accepte que 1 % d'imposteurs. C'est le réglage de la démo.

**Règles.**
- Tous les réglages (seuils, calibration, choix d'epoch, normalisation) sont faits sur la **validation**.
- Le **test** reste fermé jusqu'à l'évaluation finale.
- Toutes les approches sont comparées sur les **mêmes paires**.

**Négatives « voix proches ».** Comparer deux personnes au hasard est trop facile : un homme à voix grave contre une femme à voix aiguë. Le banc v2 ajoute donc des imposteurs **difficiles** : le quart des couples de locuteurs dont la **hauteur de voix (F0)** est la plus proche (écart de 0 à 14 Hz). La plupart des chiffres de ce rapport portent sur ces voix proches.

---

## 2. Tableau comparatif des approches

Paires de validation v1 (`pairs_val.csv`) : 710 paires (355 positives, 355 négatives), 28 locuteurs. Ce sont les mêmes paires pour toutes les lignes.

| Approche | Représentation | Score | EER | minDCF | AUC |
|---|---|---|---|---|---|
| Baseline 1 | 20 MFCC → moyenne + écart-type (40 nombres) | cosinus brut | 19,0 % | 0,676 | 0,890 |
| Baseline 2 | idem, standardisé (moyenne et écart-type du **train**) | cosinus | 11,0 % | 0,318 | 0,964 |
| Baseline 3 | paire → \|a − b\| | régression logistique | 12,3 % | 0,563 | 0,954 |
| Baseline 4 | paire → \|a − b\| et a × b | régression logistique | 8,0 % | 0,504 | 0,968 |
| Baseline 5 | paire → \|a − b\| et a × b | SVM (noyau RBF) | **6,8 %** | 0,530 | 0,963 |
| Baseline 6 | paire → \|a − b\| et a × b | MLP PyTorch 80 → 64 → 32 → 1 | 7,2 % | 0,332 | **0,976** |
| **ECAPA-TDNN** | empreinte pré-entraînée (192 nombres) | cosinus | **2,1 %** | **0,169** | **0,996** |

**Lecture.**
- **La standardisation compte énormément** pour la baseline (19,0 % → 11,0 %). Sans elle, le MFCC 0, qui mesure le volume, écrase tous les autres coefficients dans le cosinus : on compare surtout le volume d'enregistrement, pas la voix.
- **Apprendre sur les paires** aide (jusqu'à 6,8 %). Mais les classifieurs restent limités par la représentation : 40 statistiques globales ne décrivent qu'en partie l'anatomie d'une voix.
- **ECAPA est dans une autre catégorie.** Il a appris sur environ 7 000 personnes ce qui distingue les voix. Score moyen même personne : 0,688 ; personnes différentes : 0,147.

Courbes ROC : `results/roc_*.png`. Courbes d'entraînement du MLP : `results/mlp_courbes.png`.

---

## 3. Ablations

Chaque tableau ne change **qu'un facteur à la fois**, sur les mêmes paires.

### 3.1 Normalisation des empreintes ECAPA (banc v2, voix proches)

| Réglage | Entier : EER / minDCF | 3 s : EER / minDCF | 3 s : FRR@FA1% |
|---|---|---|---|
| Cosinus brut | 4,2 % / 0,225 | 4,2 % / 0,272 | 12,2 % |
| **+ soustraction de la moyenne du train** | **2,7 % / 0,141** | **2,5 % / 0,172** | **8,1 %** |
| + AS-norm | 2,3 % / 0,328 | 3,1 % / 0,337 | 27,5 % |
| + moyenne + AS-norm | 3,0 % / 0,341 | 2,7 % / 0,387 | 37,8 % |

**Décision : soustraction de la moyenne gardée, AS-norm écartée.** L'AS-norm améliore parfois l'EER, mais elle fait monter les scores de quelques imposteurs. Pour n'accepter que 1 % d'imposteurs, il faut alors refuser beaucoup plus de vraies personnes. Or c'est exactement le réglage de la démo.

**Pourquoi la soustraction aide :** toutes les empreintes partagent une composante commune (langue, type de micro, Common Voice). La retirer recentre les comparaisons sur ce qui est propre à chaque personne.

### 3.2 Durée de l'audio

Durée du **test**, avec ECAPA brut sur le banc v2 et les voix proches :

| Test | EER | minDCF |
|---|---|---|
| Clip entier (environ 5 s) | 4,2 % | 0,225 |
| 3 s | 4,2 % | 0,272 |
| 2 s | 4,4 % | 0,272 |

Durée de l'**enrôlement** (un seul bloc de parole, moyenne soustraite, test 3 s, voix proches) :

| Enrôlement | EER | minDCF | FRR@FA1% | Seuil FA1% |
|---|---|---|---|---|
| 3 s | 2,6 % | 0,126 | 3,6 % | 0,362 |
| 5 s | 2,0 % | 0,136 | 4,2 % | 0,356 |
| 8 s | 1,1 % | 0,112 | 1,0 % | 0,372 |
| 10 s | 1,0 % | 0,094 | 0,0 % | 0,378 |

- Plus l'enrôlement est long, plus le système est fiable. Mais **3 s reste très utilisable**, et le sujet récompense un enrôlement court.
- Calculer l'empreinte d'un **bloc** ou faire la **moyenne de morceaux** de 3 s donne des résultats quasi identiques. On garde le bloc, plus simple et plus proche de la démo.
- **2 s** est abandonné : la dégradation est trop forte pour un gain de temps négligeable.
- **Choix : 3 s.**

### 3.3 Bruit (banc v2, enrôlement 3 s propre, test 3 s bruité, voix proches, seuil 0,362)

| Condition | EER | minDCF | Vraie personne refusée | Imposteur accepté |
|---|---|---|---|---|
| Propre | 2,5 % | 0,172 | 5,9 % | 1,9 % |
| Ambiance 20 / 10 / 5 dB | 2,2 / 3,1 / 4,7 % | 0,188 / 0,228 / 0,316 | 7,5 / 11,2 / 20,0 % | 1,6 / 1,2 / 0,9 % |
| Brouhaha 20 / 10 / 5 dB | 3,3 / 7,8 / 11,9 % | 0,162 / 0,337 / 0,447 | 7,8 / 23,1 / 41,6 % | 0,9 / 0,9 / 0,3 % |

- « Ambiance » est un bruit rose continu (ventilation, salle). « Brouhaha » est fait de 4 voix de locuteurs de train superposées.
- **Le bruit fait baisser tous les scores.** On accepte donc *moins* d'imposteurs, mais on refuse *beaucoup plus* de vraies personnes. Le système reste prudent.
- **Le brouhaha est le pire**, parce qu'il est fait de **voix** : il brouille précisément ce que le modèle écoute.
- **Conséquence pour la démo :** salle calme, micro proche, et test jusqu'à 10 s (plus de voix = décision plus sûre).

### 3.4 Préparation de la voix : banc v2 → banc v3

Le banc v2 prenait le **milieu du clip brut**, silences compris. La démo, elle, **retire tous les silences et les pauses**, puis garde les N premières secondes **de parole**. Les seuils mesurés sur le banc v2 ne correspondaient donc pas exactement aux conditions du jour J.

Le **banc v3** reprend les **mêmes 960 paires** et prépare chaque voix avec **la même fonction que la démo** (`src/preparation.py`) :
- enrôlement de 3, 5 ou 10 s de parole, en un bloc, sans jamais utiliser le clip de test ;
- test de 3 s ou jusqu'à 10 s ;
- bruit ajouté **avant** le retrait des silences, comme dans une vraie salle.

**Quelle moyenne soustraire ?** (enrôlement 3 s, voix proches, moyenne sur 6 tests : 2 durées × 3 conditions)

| Moyenne soustraite | minDCF moyen | FRR@FA1% moyen |
|---|---|---|
| Train cv26, clips bruts (ancienne) | 0,222 | 8,9 % |
| **Train cv26, préparée comme la démo** | **0,217** | **8,2 %** |
| cv27, 945 locuteurs | 0,235 | 10,5 % |

La moyenne fabriquée avec la même recette que la démo est la meilleure (dans 5 tests sur 6). Celle de cv27, venue d'autres personnes et d'autres conditions, **dégrade** les résultats. C'est un avertissement utile : une moyenne doit ressembler aux voix qu'elle normalise.

**Résultats du banc v3** (moyenne « cv26 préparée démo », voix proches)

| Enrôlement | Test | Condition | EER | minDCF | FRR@FA1% | Seuil FA1% |
|---|---|---|---|---|---|---|
| 3 s | 3 s | propre | 2,2 % | 0,175 | 5,6 % | 0,416 |
| 3 s | ≤ 10 s | **propre** | **1,9 %** | **0,172** | **5,3 %** | 0,416 |
| 3 s | ≤ 10 s | ambiance 10 dB | 2,8 % | 0,194 | 5,0 % | 0,343 |
| 3 s | ≤ 10 s | brouhaha 10 dB | 5,9 % | 0,241 | 11,3 % | 0,298 |
| 5 s | ≤ 10 s | propre | 2,0 % | 0,156 | 3,7 % | 0,447 |
| 10 s | ≤ 10 s | propre | 1,9 % | 0,153 | 3,4 % | 0,482 |

Tableau complet : `results/banc_v3.csv`. Quelques personnes de validation n'ont pas assez de parole pour 10 s : 21 enrôlements sur environ 2 000 sont plus courts que prévu.

---

## 4. Metric learning

**Idée.** Ajouter au-dessus d'ECAPA gelé une **petite couche apprise**, qui rapproche les empreintes d'une même personne et éloigne celles de personnes différentes, dans les conditions de la démo : extraits de 3 s de phrases différentes, avec ou sans bruit. La couche part de « ne rien changer » : à l'epoch 0, on retrouve exactement ECAPA. Elle n'est gardée **que si elle améliore la validation**.

### 4.1 Version 1 : triplet loss, 85 locuteurs

- Couche résiduelle 192 → 192 (environ 37 000 paramètres).
- **Triplet loss** « batch-hard », marge 0,2. Dans chaque lot de 16 personnes × 6 empreintes, chaque ancre est comparée à son positif le plus éloigné et à son négatif le plus proche.
- Données : 85 locuteurs de train cv26 × 6 versions (2 propres, ambiance et brouhaha à 20 et 10 dB), soit 6 030 empreintes. 40 epochs.

| Voix proches, 3 s / 3 s | minDCF ECAPA → + couche | FRR@FA1% ECAPA → + couche |
|---|---|---|
| Propre | 0,172 → 0,188 | 8,1 % → 9,1 % |
| Ambiance 10 dB | 0,225 → 0,262 | 12,8 % → 10,0 % |
| Brouhaha 10 dB | 0,266 → 0,206 | 12,2 % → 19,4 % |

**Diagnostic : overfit.** La loss d'entraînement est déjà très basse à l'epoch 5 (0,07) et continue de baisser, pendant que la validation se dégrade (minDCF propre 0,172 → 0,244 à l'epoch 35). Les causes identifiées sont :
1. trop peu de personnes (85, contre environ 7 000 pour ECAPA) ;
2. une triplet loss trop facile ;
3. une couche trop libre ;
4. un critère de choix d'epoch instable.

Courbes : `results/metric_learning_courbes.png`.

### 4.2 Version 2 : AAM-softmax, 945 locuteurs

Chaque cause est corrigée :
1. **945 locuteurs** de Common Voice 27.0, équilibrés par hauteur de voix, sans aucun locuteur de validation ni de test cv26. 94 d'entre eux sont **tenus à l'écart** pour surveiller l'overfit.
2. **AAM-softmax** (marge 0,2, échelle 30), la loss d'entraînement d'ECAPA.
3. **Deux couches contraintes**, rappelées vers « ne rien changer » par une pénalité : *diagonale* (un poids par dimension, 192 paramètres) et *résiduelle de rang 32* (12 480 paramètres).
4. **Critère stable** : minDCF moyen sur toutes les paires de 3 s (propre, ambiance et brouhaha à 10 dB), lissé sur 3 epochs.

| Voix proches, 3 s / 3 s | ECAPA | + diagonale | + résiduelle r32 |
|---|---|---|---|
| minDCF propre / ambiance / brouhaha | 0,172 / 0,225 / 0,266 | 0,178 / 0,219 / 0,228 | identique à ECAPA (epoch 0) |
| FRR@FA1% propre / ambiance / brouhaha | 8,1 / 12,8 / 12,2 % | 9,1 / 14,7 / 12,2 % | identique à ECAPA |

**Ce que montre la surveillance :** la couche résiduelle **progresse sur les 94 personnes jamais vues** de cv27 (EER propre 3,3 % → 2,4-3,0 % ; brouhaha 5,6 % → 4,8 %). **L'overfit sur les personnes est donc corrigé.** Mais ce gain **ne se transfère pas** au banc de validation cv26.

Re-testées sur le banc v3, à préparation identique, les couches ne font pas mieux non plus : la diagonale gagne un peu en brouhaha et perd en ambiance (FRR@FA1% 9,1 % → 10,9 %).

**Conclusion.** Avec une couche légère au-dessus d'un modèle déjà entraîné sur des milliers de personnes, il ne reste pas de gain **fiable** à prendre sur nos données. Le metric learning n'est **pas intégré** à la démo. La piste a été explorée jusqu'au bout : la cause de l'échec de la v1 a été trouvée et corrigée, ce qui permet de conclure sur des preuves. Courbes : `results/metric_learning_v2_courbes.png`.

---

## 5. Partie explicable : qu'est-ce qui distingue deux voix ?

**Caractéristiques nommées** (Praat et librosa, `src/caracteristiques.py`) : hauteur (F0), variation de hauteur, formants F1 à F3, jitter, shimmer, HNR, centroïde spectral, rythme, MFCC 1 à 12.

**Ratio de Fisher** (variation *entre* personnes / variation *chez une même* personne, calculé sur le train) :

| Caractéristique | Ratio | Ce qu'elle décrit |
|---|---|---|
| Hauteur moyenne (F0) | **10,6** | voix grave ou aiguë : taille et tension des cordes vocales |
| MFCC (timbre) | 1,6 à 3,9 | forme du spectre : résonances de la gorge et de la bouche |
| HNR | 2,4 | voix claire ou voilée |
| Shimmer | 2,0 | irrégularités de volume |
| Formant F3 | 1,9 | longueur et forme du conduit vocal |
| Formant F1 | 1,55 | ouverture de la bouche |
| Jitter | 1,0 | irrégularités de hauteur |
| Variation de hauteur | 0,79 | intonation |
| Formant F2 | 0,69 | position de la langue |
| Rythme | 0,51 | vitesse de parole |

**Interprétation.** L'identité vient surtout de l'**anatomie** : la hauteur, les résonances et la clarté de la voix. Elle vient beaucoup moins de la **façon de parler**. F2, le rythme et l'intonation dépendent surtout de la phrase et des voyelles prononcées, et changent d'un enregistrement à l'autre chez une même personne.

C'est cohérent avec la demande du mentor : reconnaître une personne par son **timbre**, quels que soient la langue et les mots.

**Plus proche voisin** (282 clips de validation, 28 personnes, hasard ≈ 4 %) : le clip le plus ressemblant vient-il de la même personne ?

| Représentation | Taux de réussite |
|---|---|
| Caractéristiques nommées seules | 63 % |
| + MFCC | 93 % |
| **ECAPA** | **98 %** |

Les dimensions d'ECAPA prises **une par une** ont un ratio de Fisher moyen de 1,76 (maximum 3,04) : aucune ne « sait » tout. C'est leur **combinaison** qui fait la force du modèle.

**Carte t-SNE** (`results/tsne_ecapa.png`) : chaque îlot correspond à une personne. Les 9 prises de l'auteur (français, espagnol, anglais, fausse langue, loin du micro) forment **un seul groupe serré**. C'est une preuve visuelle que l'empreinte capte **la voix, pas la langue**.

**Dans la démo**, le panneau « Pourquoi ? » affiche les 3 caractéristiques nommées qui diffèrent le plus entre la voix enrôlée et la voix testée. Chaque écart est exprimé en « variations habituelles chez une même personne ».

---

## 6. Calibration et seuil

**Seuil.** Il est choisi pour n'accepter que **1 % d'imposteurs parmi les voix proches** (validation, banc v3). On prend le plus prudent des deux tests (3 s ou jusqu'à 10 s), pour tenir cette garantie que la personne parle peu ou beaucoup.

| Enrôlement | 3 s | 5 s | 10 s |
|---|---|---|---|
| Seuil | **0,416** | 0,447 | 0,482 |

**Calibration.** Une régression logistique (classes équilibrées, donc « une chance sur deux » au départ) transforme le score en probabilité :
p = 1 / (1 + exp(−(10,54 × score − 3,38))). Au seuil de 3 s, la probabilité vaut **73 %**.

Elle a été vérifiée en apprenant sur la moitié des locuteurs et en testant sur l'autre moitié :

| Probabilité annoncée | Paires | Vraie personne en réalité |
|---|---|---|
| 0 à 10 % | 110 | 0 % |
| 10 à 50 % | 36 | 25 % |
| 50 à 90 % | 79 | 96 % |
| 90 à 100 % | 91 | **99 %** |

**Affichage dans la démo.**
- ✅ Personne acceptée : décision en grand, avec la confiance calibrée.
- ❌ Personne refusée : décision **sans pourcentage**, pour éviter un « 65 % » affiché à côté d'un « non ».
- Dans les deux cas, le score brut et le seuil sont affichés en petit.

---

## 7. Démonstration

**Configuration figée.**
- ECAPA gelé, avec soustraction de la moyenne « cv26 préparée démo ».
- Enrôlement de **3 s de parole** (un bloc, silences retirés) ; 5 s et 10 s sont aussi proposés.
- Test jusqu'à **10 s de parole**.
- Seuils et calibration du banc v3.

**Parcours.** Consentement → enrôlement → passages enchaînés, avec un historique des décisions → explication « Pourquoi ? » → suppression de l'empreinte.

**Tests en conditions réelles.**
- **Démo v0** (enrôlement de 20 s) : 10 essais de l'auteur, tous corrects. Scores : français 0,66-0,77, espagnol 0,65, anglais 0,62, loin du micro 0,69, fausse langue 0,77. Ses proches : 0,08-0,23.
- **Mini-corpus** (enrôlement 3 s, ancien seuil de 0,362) : 12 essais sur 12 de la vraie personne acceptés (scores de 0,382 à 0,627), 0 proche accepté sur 12 (score maximum 0,088), 2 imposteurs Common Voice acceptés sur 564.
- **Démo v1, enrôlement 3 s** : quelques refus quand l'enrôlement est en anglais ou en espagnol et le test en français (scores de 0,230 et 0,341). Après le retrait des pauses internes, 4 essais anglais → français ont été acceptés (0,39 à 0,49).
- **Démo avec les réglages du banc v3 (seuil 0,416)** : [À COMPLÉTER : résultats du plan de test, avec un enrôlement en français puis en anglais, des passages en 3 langues, loin du micro, et d'autres personnes]

---

## 8. Cas d'échec

**Échecs connus, mesurés.**
1. **Changement de langue avec un enrôlement très court.** En 3 s, l'empreinte dépend en partie des **sons** prononcés, et chaque langue a les siens. Un enrôlement en anglais ou en espagnol suivi d'un test en français fait baisser le score près du seuil, voire en dessous. Le sens inverse (enrôlement en français, test dans une autre langue) passe mieux.
2. **Brouhaha.** À 10 dB, 11 à 23 % des vraies personnes sont refusées selon le banc ; à 5 dB, 42 %. Des voix parasites brouillent ce que le modèle écoute.
3. **Distance au micro.** Les scores baissent (0,40-0,42 sur le mini-corpus à environ 1,5 m), tout en restant au-dessus du seuil de l'époque.
4. **Voix proches.** Les rares imposteurs acceptés sont des personnes de **même hauteur de voix**. C'est pour cela que les seuils sont réglés sur les voix proches.

**Analyse des pires paires de validation.** [À COMPLÉTER : écouter les 5 vraies personnes refusées avec les scores les plus bas et les 5 imposteurs acceptés avec les scores les plus hauts (banc v3, enrôlement 3 s), puis noter la cause probable de chacun : bruit, voix chuchotée, clip très court, micro saturé, voix très proches…]

---

## 9. Limites et attaques

**Limites de l'évaluation**
- Tous les chiffres viennent de **français lu**. La démo utilisera de la parole spontanée, en plusieurs langues.
- Le banc de validation est de taille modeste (320 positives, 320 voix proches) : **±1 point environ** sur les taux, et un seuil à 1 % d'imposteurs qui repose sur environ 3 paires.
- Le mini-corpus ne compte que 2 personnes : il illustre, il ne prouve pas.
- Les choix d'epoch et de moyenne ont été faits sur la validation, dont les chiffres sont donc légèrement optimistes. Le **test final** donnera le chiffre non biaisé : [À COMPLÉTER].
- **Voix d'enfants** non mesurées : ECAPA a été entraîné sur des adultes, et les performances sont probablement plus faibles.
- Le bruit est simulé, pas enregistré.

**Attaques (non traitées).** Le système vérifie que la voix *ressemble* à celle de la personne enrôlée. Il ne vérifie **pas** qu'elle est *vivante* :
- **Rejeu** : un enregistrement de la personne, joué sur un téléphone, sera très probablement accepté. [À COMPLÉTER si le test de rejeu est fait.]
- **Voix synthétique ou clonée** : non détectée. La détection d'usurpation (*anti-spoofing*, voir le challenge ASVspoof) est un problème à part entière.
- **Imitation** par une personne de voix proche : partiellement couverte par le seuil réglé sur les voix proches.

**C'est pourquoi la démo n'est pas un dispositif d'authentification sécurisé.**

---

## 10. Éthique et données biométriques

- **Consentement** : il est demandé explicitement avant tout enrôlement dans la démo et avant tout enregistrement du mini-corpus.
- **Minimisation** :
  - la démo ne garde **que l'empreinte** (192 nombres) et les caractéristiques nommées, jamais l'audio ;
  - le genre n'est pas collecté ;
  - on ne garde ni l'âge, ni le genre, ni l'accent, ni le texte des corpus Common Voice ;
  - l'archive de 28 Go a été supprimée après usage.
- **Suppression** : un bouton efface l'empreinte de la démo. Le mini-corpus sera supprimé à la fin du projet ou sur demande.
- **Pas de redistribution** : aucun audio ni aucune empreinte sur GitHub. Le corpus Common Voice n'est pas re-partagé, et aucune tentative d'identifier ses locuteurs n'est faite, conformément à sa licence.
- **Usage honnête** : le système est présenté comme une démonstration pédagogique, avec ses limites et ses attaques possibles.

---

## 11. Bilan et suites possibles

**Ce qui marche.** Des empreintes pré-entraînées, une normalisation simple et un seuil réglé avec soin sur des imposteurs difficiles. Le tout est mesuré dans les conditions exactes de la démo.

**Ce qui n'a pas marché, et pourquoi.**
- L'AS-norm : elle accepte des imposteurs à score élevé.
- Le metric learning : la v1 a sur-appris ; la v2 a été corrigée mais ne donne aucun gain fiable.
- La moyenne cv27 : elle ne correspond pas à la population évaluée.

**Suites possibles.**
- Mesurer et corriger la sensibilité au changement de langue, par exemple avec un enrôlement de 5 s ou une moyenne multilingue (à tester avec prudence, voir 3.4).
- Comparer une deuxième famille de modèles (WeSpeaker).
- Enregistrer un vrai banc de bruit et de parole spontanée.
- Ajouter une détection de rejeu.
