# Classification binaire sur Spambase et BRFSS 2015

Ce projet applique le même workflow à deux décisions différentes. Pour un filtre anti-spam, bloquer un message légitime coûte cher. Pour l’enquête diabète, la première question est de retrouver une classe positive rare. Les notebooks choisissent donc leurs critères avant l’entraînement. [Sujet du projet](description.pdf)

| Notebook | Données livrées | Critère de sélection | Résultat sur le test réservé |
|---|---|---|---|
| [SPAM](spam.ipynb) | 4 601 e-mails, 57 variables | F₀,₅ sur des profils inédits | Boosting, 300 spams détectés sur 363 et 5 messages légitimes bloqués au seuil ajusté |
| [Diabète](dhi.ipynb) | 253 680 répondants, 21 variables | Précision moyenne (AP) | Boosting, AP de 0,425 pour une prévalence de 0,139 |

## Reproduire le projet

Depuis la racine du dépôt, avec Python 3.12, exécuter les commandes suivantes. Les deux fichiers de données utilisés sont inclus dans `datasets/`. Le chemin SPAM explicite permet de rejouer le notebook hors ligne après l’installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests.py
jupyter notebook
```

Dans Jupyter, choisir le noyau de l’environnement et exécuter toutes les cellules dans l’ordre. Pour exécuter les deux notebooks en ligne de commande et enregistrer une nouvelle copie de leurs sorties dans un répertoire distinct :

```bash
mkdir -p /tmp/projet-ml-results
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1500 \
  --output-dir=/tmp/projet-ml-results spam.ipynb dhi.ipynb
```

## Protocole et lecture des résultats

Toute la logique d’import, de nettoyage, de séparation, de transformation, d’apprentissage et de calcul des scores se trouve dans [ml_workflow.py](ml_workflow.py). Les notebooks configurent l’expérience, présentent ses résultats et les interprètent. Les graphiques descriptifs initiaux portent sur le corpus. Les classements de variables, l’ajustement des transformations et la recherche de modèles utilisent exclusivement l’entraînement.

- **Unité d’observation.** Les répétitions sont comptées et conservées. Pour SPAM, les profils de variables identiques sont regroupés lors du découpage final et de la validation croisée. Pour le diabète, une séparation stratifiée par ligne reproduit la fréquence de la cible.
- **Comparaison équitable.** Chaque famille utilise les mêmes cinq plis sur environ 80 % des observations. La régression logistique compare les variables standardisées, une sélection par test F et une PCA conservant 95 % de la variance. Un arbre de décision compare deux profondeurs. Le boosting compare deux nombres maximaux de feuilles. Chaque meilleur candidat est réajusté sur tout l’entraînement. Le choix de la famille précède l’examen des scores finaux.
- **Interprétation.** La référence majoritaire révèle ce qu’un score d’exactitude élevé peut masquer. Pour SPAM, une seconde règle maximise le rappel sous une contrainte de précision empirique de 98 % sur les prédictions hors pli du boosting. Le seuil est gelé avant le calcul des erreurs de cette règle sur le test. Pour le diabète, les poids de classe font partie de la grille et la comparaison isole leur effet sur deux boostings de même complexité. Les courbes précision-rappel et ROC décrivent le classement lorsque le seuil varie. L’AP est une moyenne pondérée des précisions aux différents niveaux de rappel.
- **Portée.** La dispersion entre plis décrit la sensibilité au découpage. Les deux tests sont internes aux corpus. La version précédente de cet exemple avait déjà publié les scores aux seuils par défaut sur ces découpages. Les nouvelles analyses réutilisent les mêmes tests pour permettre une comparaison appariée. Seules les données d’entraînement servent à calculer les nouveaux réglages. Le filtrage de courriels contemporains et la prédiction dans une autre population nécessiteraient des données externes.

## Points d’appui pour la correction

Une autre copie peut défendre un autre modèle. Pour comparer les travaux, vérifier d’abord l’unité de test, le sens de la classe positive et le protocole de sélection : ces choix déterminent ce que mesure chaque score.

| Question à poser | Élément vérifiable dans le projet |
|---|---|
| Une observation de test intervient-elle dans le choix du modèle ? | Le classement en validation croisée fixe le gagnant avant le tableau de test. Les transformations sont dans le pipeline ajusté à chaque pli. |
| La réduction de dimension apporte-t-elle quelque chose ? | Les trois représentations de la régression sont comparées sur les mêmes plis avec la même grille de régularisation. |
| Le traitement propre à chaque jeu change-t-il le résultat ? | SPAM compare deux découpages et chiffre le coût d’un blocage plus strict. Diabète compare les pondérations à structure de boosting et plis fixés. |
| La conclusion correspond-elle à l’usage ? | Chaque notebook confronte le critère principal aux faux positifs, aux faux négatifs et aux limites du corpus. |
| Le code vérifie-t-il ses hypothèses ? | Les [tests](tests.py) couvrent notamment l’alignement des cibles, les doublons, l’isolement des groupes, les statistiques apprises sur l’entraînement et des métriques calculables à la main. |

## Données et assistance à la réalisation

Spambase est un corpus constitué chez Hewlett-Packard en 1999. Les variables décrivent des fréquences et des suites de majuscules. [Fiche UCI](https://archive.ics.uci.edu/dataset/94/spambase) · [Documentation livrée avec les données](datasets/spam/spambase.DOCUMENTATION)

Le fichier diabète est la version binaire non rééquilibrée des *Diabetes Health Indicators*, dérivée de l’enquête américaine BRFSS 2015. [Distribution et description des fichiers](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

Claude Code a fourni le squelette initial du module, des tests et des notebooks. Codex a servi à compléter le workflow commun, à construire les comparaisons et à rédiger les interprétations de cet exemple pédagogique. Les résultats ont été contrôlés en exécutant les notebooks sur les fichiers du dépôt et en testant séparément les règles de préparation et de calcul des scores.
