# Consignes universitaires

[Retour au README](../README.md)

### Étape 1 — Choix du problème

House Prices a été sélectionnée. Vérifier que les données sont libres d'accès, téléchargeables sans restriction particulière et adaptées au cours. Kaggle peut demander un compte et l'acceptation des règles : documenter ces conditions et vérifier leur conformité aux exigences universitaires.

### Étape 2 — Présentation du problème

Le lien, la problématique et l'objectif figurent dans le [README](../README.md). Compléter dans le premier notebook la description des variables, types, unités et catégories à partir de `data_description.txt`. Présenter les défis : valeurs manquantes, catégories, asymétrie des prix, valeurs atypiques, corrélations et généralisation; les confirmer par l'analyse.

### Étape 3 — Analyse exploratoire

Dans `notebooks/01_exploration.ipynb` : description des données, statistiques descriptives, identification des valeurs manquantes, distributions, corrélations et visualisations pertinentes. **Chaque figure devra être clairement commentée et interprétée** : observation, signification et conséquence pour la suite. Une corrélation ne démontre pas une causalité.

### Étape 4 — Prétraitement

Dans `notebooks/02_pretraitement.ipynb`, décrire et justifier nettoyage, traitement des valeurs manquantes, encodage des catégories, normalisation ou standardisation, sélection et création de caractéristiques (*feature engineering*). Les futures fonctions communes iront dans `src/preprocessing.py`. Prévoir des pipelines scikit-learn : ajuster les transformations apprises sur l'entraînement uniquement, y compris dans chaque pli de validation croisée.

### Étape 5 — Développement de la solution

Dans `notebooks/03_modelisation_evaluation.ipynb`, identifier le **modèle principal : à choisir ultérieurement**, ses paramètres, hypothèses et justifications techniques. Des comparaisons seront possibles en distinguant clairement le modèle retenu. Les fonctions communes iront dans `src/training.py`. Aucun modèle n'est encore implémenté.

### Étape 6 — Évaluation

Prévoir les fonctions futures dans `src/evaluation.py` et présenter :

| Métrique | Signification |
|---|---|
| MAE | Erreur absolue moyenne en dollars; plus faible est meilleur. |
| MSE | Erreur quadratique moyenne en dollars carrés; accentue les grandes erreurs. |
| RMSE | Racine de la MSE en dollars; sensible aux erreurs importantes. |
| R² | Part de variabilité expliquée; 1 est idéal, une valeur négative est possible. |
| RMSE des logarithmes | Mesure officielle Kaggle, distincte des erreurs en dollars. |

La [compétition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) utilise la racine de la moyenne de `(log(prix_prédit) - log(prix_observé))²`, sur des prix strictement positifs. Cette métrique est souvent appelée RMSLE dans ce contexte; attention : `root_mean_squared_log_error` de scikit-learn utilise `log1p`, ce qui n'est pas exactement le même calcul. Documenter explicitement la formule utilisée.

Évaluer localement sur une validation séparée; comparer les modèles sur les mêmes partitions. Présenter un tableau des métriques et leur interprétation. Les vrais prix de `test.csv` ne sont pas disponibles.

Si possible : générer un CSV conforme à `sample_submission.csv`, soumettre sur Kaggle, conserver une capture d'écran dans `docs/`, noter score, date et position approximative au classement. Comparer et expliquer les résultats locaux et Kaggle. Si aucune soumission n'est possible, justifier cette situation et fournir l'évaluation locale. Éviter d'ajuster continuellement le modèle au classement public.

### Étape 7 — Analyse critique

Compléter le troisième notebook : performances, limites, erreurs fréquentes ou importantes, améliorations possibles et pistes futures. Appuyer les conclusions sur les résultats obtenus.

## Livrables universitaires

1. **PowerPoint de 10 à 15 diapositives** : introduction, description du problème, analyse exploratoire, prétraitement, modélisation, évaluation, analyse critique et conclusion. Préparer également son export PDF.
2. **Code source** : notebooks Jupyter ou scripts Python exécutables et commentés, avec dépendances et instructions.
3. **Résultats** : prédictions soumises si applicable, captures Kaggle et tableau récapitulatif des métriques interprétées.

La remise électronique sur la plateforme du cours comprend la présentation en **PDF**, le code et les résultats d'évaluation. Les soumissions étant ignorées par Git, inclure explicitement les CSV nécessaires dans le dossier de remise.

## Échéances

- Remises mentionnées : **2 décembre 2026** et **9 décembre 2026**.
- Présentations : **9 décembre 2026**.
- Attribution des deux dates aux éléments à remettre : **À confirmer avec l'enseignant**.

## Barème d'évaluation

| Critère | Points |
|---|---:|
| Choix et compréhension du problème | 10 |
| Analyse exploratoire des données | 20 |
| Prétraitement des données | 15 |
| Développement des modèles | 20 |
| Évaluation et métriques | 15 |
| Analyse critique | 10 |
| Qualité de la présentation | 10 |
| **Total** | **100** |

**Bonus potentiel : +10 points**, sans garantie. Les consignes mentionnent la création originale de nouveaux concepts propres aux étudiants et l'obtention d'un score compétitif au classement Kaggle. Les modalités restent à valider.
