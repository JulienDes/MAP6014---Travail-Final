# Architecture du projet

[Retour au README](../README.md)

## Organisation proposée

Cette structure est une cible pour la suite du projet, pas un inventaire des fichiers actuels. Les noms et les modules pourront évoluer selon les expérimentations retenues. Les étapes 4 et 5 seront développées en Python; les notebooks serviront à présenter le problème, explorer les données et discuter les résultats.

```text
house-prices/
├── data/
├── notebooks/
│   ├── 01_choix_probleme.ipynb
│   ├── 02_presentation_probleme.ipynb
│   ├── 03_analyse_exploratoire.ipynb
│   ├── 04_pretraitement.ipynb
│   ├── 05_developpement_solution.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_analyse_critique.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pipelines.py
│   │   ├── steps/
│   │   │   ├── __init__.py
│   │   │   ├── cleaning.py
│   │   │   ├── missing.py
│   │   │   ├── features.py
│   │   │   ├── encoding.py
│   │   │   └── scaling.py
│   │   └── strategies/
│   │       ├── __init__.py
│   │       ├── correlation.py
│   │       ├── causal.py
│   │       └── pca.py
│   ├── training.py
│   └── evaluation.py
├── submissions/
├── docs/
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md
```

| Élément | Rôle |
|---|---|
| `data/` | Fichiers Kaggle locaux, ignorés par Git; données brutes à garder intactes. |
| `notebooks/` | Présentation et analyses, avec une numérotation correspondant aux étapes universitaires. |
| `src/preprocessing/pipelines.py` | Assemble cinq recettes pour l'étape 4 (base, corrélation, causale, ACP, causale avec ACP par blocs). |
| `steps/cleaning.py` | Harmonise les codes et traite les incohérences. Permet d'exclure les ventes atypiques de l'entraînement. |
| `steps/missing.py` | Distingue les absences d'équipement des valeurs inconnues à imputer. |
| `steps/features.py` | Crée les caractéristiques dérivées, dont l'âge à la vente. |
| `steps/encoding.py` | Encode les qualités ordinales et les catégories nominales. |
| `steps/scaling.py` | Corrige l'asymétrie par `log1p` et standardise les caractéristiques. |
| `strategies/correlation.py` | Sélectionne des variables selon leur corrélation avec le prix. |
| `strategies/causal.py` | Sélectionne des variables selon les hypothèses causales retenues. |
| `strategies/pca.py` | Réduit la dimension par ACP globale ou par blocs de variables. |
| `src/training.py` | Charge les données, construit le modèle et évalue le pipeline sur une partition réservée ou par validation croisée (étape 5). |
| `src/evaluation.py` | Calcule les métriques de régression et compare les résultats pour l'étape 6. |
| `submissions/` | Prédictions générées, ignorées par Git. |
| `docs/` | Références, présentation, captures et bilans. |

Les fichiers de configuration à la racine définissent les dépendances, la version Python et les exclusions Git. Le README renvoie aux guides détaillés; les `.gitkeep` permettent de conserver les dossiers vides.

Les dossiers `steps/` et `strategies/` se trouvent dans `src/preprocessing/`. Chaque recette sera ajustée uniquement sur les maisons d'entraînement de la partition ou du pli en cours.
