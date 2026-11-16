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
│   ├── 06_evaluation.ipynb
│   └── 07_analyse_critique.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   ├── training.py
│   └── evaluation.py
├── submissions/
├── docs/
├── tests/
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md
```

| Élément | Rôle |
|---|---|
| `data/` | Fichiers Kaggle locaux, ignorés par Git; données brutes à garder intactes. |
| `notebooks/` | Présentation et analyses, avec une numérotation correspondant aux étapes universitaires. |
| `src/preprocessing/` | Étape 4 : plusieurs modules de prétraitement, de sélection ou de transformation des variables, selon les besoins. |
| `src/training.py` | Étape 5 : entraînement et comparaison de différents modèles. |
| `src/evaluation.py` | Étape 6 : métriques et prédictions; résultats présentés dans le notebook 06. |
| `submissions/` | Prédictions générées, ignorées par Git. |
| `docs/` | Références, présentation, captures et bilans. |
| `tests/` | Futurs tests des fonctions réutilisables. |

Les fichiers de configuration à la racine définissent les dépendances, la version Python et les exclusions Git. Le README renvoie aux guides détaillés; les `.gitkeep` permettent de conserver les dossiers vides.

Le découpage des prétraitements reste libre : ajouter, regrouper ou retirer des modules selon les approches réellement explorées, sans imposer un fichier pour chaque méthode envisagée.
