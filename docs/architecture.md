# Architecture du projet

[Retour au README](../README.md)

```text
house-prices/
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_pretraitement.ipynb
│   └── 03_modelisation_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluation.py
├── submissions/
│   └── .gitkeep
├── docs/
│   ├── .gitkeep
│   ├── utilisation.md
│   ├── architecture.md
│   ├── collaboration.md
│   ├── consignes-universitaires.md
│   └── travail-de-mi-session-v2.pdf
├── tests/
│   └── .gitkeep
├── .gitignore
├── .python-version
├── pyproject.toml
└── README.md
```

| Élément | Rôle |
|---|---|
| `data/` | Fichiers Kaggle locaux, ignorés par Git; données brutes à garder intactes. |
| `notebooks/` | Analyses documentées et expérimentations, dans l'ordre numérique. |
| `src/__init__.py` | Déclaration du paquet réutilisable. |
| `src/preprocessing.py` | Futures fonctions de nettoyage, caractéristiques et pipelines. |
| `src/training.py` | Futurs entraînements et validation croisée. |
| `src/evaluation.py` | Futures métriques, comparaisons et soumissions. |
| `submissions/` | Prédictions générées, ignorées par Git. |
| `docs/` | Références, présentation, captures et bilans. |
| `tests/` | Futurs tests des fonctions réutilisables. |
| `.gitkeep` | Conservation des dossiers vides dans Git. |
| `.gitignore` | Exclusions des données et fichiers locaux. |
| `.python-version` | Version mineure Python retenue. |
| `pyproject.toml` | Métadonnées, dépendances et construction du paquet. |
| `README.md` | Présentation sommaire et liens vers les guides. |

Emplacement prévu des consignes officielles : `docs/travail-de-mi-session-v2.pdf`. Son ajout ou sa vérification par l'équipe reste une tâche documentaire. Ce guide reprend les consignes fournies lors de l'initialisation, sans validation du PDF. Les modules ne contiennent que des docstrings de réservation.
