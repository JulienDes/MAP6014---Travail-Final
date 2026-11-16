# Installation et utilisation

[Retour au README](../README.md)

## Démarrage rapide — Windows / PowerShell

Il faut **Python 3.14 standard, 64 bits** et **VS Code** avec les extensions **Python** et **Jupyter**. Si Python manque, l'installer depuis [python.org](https://www.python.org/downloads/), puis rouvrir VS Code. La commande `py -3.14 --version` doit afficher `Python 3.14...`.

### 1. Créer `.venv` et installer les dépendances

Dans VS Code, ouvrir le dossier du projet, puis **Terminal > Nouveau terminal**. Le terminal doit être à la racine du projet, dans le dossier contenant `pyproject.toml`.

Copier ces trois commandes dans PowerShell :

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

- La première crée l'environnement dans le dossier `.venv`.
- La deuxième l'active pour ce terminal : les commandes `python` utilisent alors cet environnement.
- La troisième installe le code du projet et toutes les dépendances déclarées dans `pyproject.toml`. La première installation peut prendre quelques minutes.

Si PowerShell bloque l'activation, voir [le dépannage](#en-cas-de-problème) plus bas.

Pour vérifier que le bon Python est utilisé :

```powershell
python -c "import sys; print(sys.executable)"
```

Le chemin affiché doit finir par `.venv\Scripts\python.exe`.

### 2. Utiliser les notebooks dans VS Code

1. Faire `Ctrl+Shift+P`, choisir **Python: Select Interpreter**, puis `.venv\Scripts\python.exe`.
2. Ouvrir un notebook et cliquer sur **Select Kernel** en haut à droite.
3. Choisir **Python Environments**, puis le Python de `.venv`.
4. Cliquer sur **Run All** pour exécuter toutes les cellules, ou sur le bouton ▶ d'une cellule pour l'exécuter seule.

`ipykernel`, nécessaire aux notebooks, est déjà installé à l'étape 1. Les indications sur les données et l'ordre des notebooks se trouvent dans la section [Utilisation](#utilisation).

### 3. Lancer un script Python

Dans le terminal avec `.venv` activé, lancer un fichier Python avec :

```powershell
python chemin\vers\script.py
```

Remplacer `chemin\vers\script.py` par le chemin du fichier à exécuter, depuis la racine du projet. C'est un exemple pour les futurs scripts : le projet ne contient pas encore de programme principal. Les modules de `src` peuvent être importés dans les notebooks, par exemple avec `from src import preprocessing`.

### 4. Reprendre le travail une autre journée

Ouvrir le dossier du projet dans VS Code, puis activer l'environnement dans le nouveau terminal :

```powershell
.\.venv\Scripts\Activate.ps1
```

Pour les notebooks, vérifier que le kernel sélectionné est toujours `.venv`. **Il n'est pas nécessaire de recréer l'environnement ni de réinstaller les dépendances à chaque séance.**

Si les dépendances de `pyproject.toml` changent, relancer `python -m pip install -e .` dans le terminal activé. Pour désactiver l'environnement à la fin d'une séance, taper `deactivate` ou fermer le terminal.

## En cas de problème

### PowerShell bloque l'activation

Autoriser les scripts uniquement pour le terminal en cours, puis réessayer :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Le kernel `.venv` n'apparaît pas ou une bibliothèque est introuvable

Vérifier que l'installation de l'étape 1 s'est terminée sans erreur. Dans VS Code, faire `Ctrl+Shift+P`, puis **Developer: Reload Window**. Refaire la sélection de l'interpréteur et du kernel décrite à l'étape 2, puis redémarrer le kernel.

## Utilisation

1. Consulter [Data sur Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), vérifier les règles et télécharger manuellement les fichiers autorisés.
2. Décompresser directement dans `data/` : `train.csv`, `test.csv`, `sample_submission.csv` et `data_description.txt`.
3. Ouvrir [le choix du problème](../notebooks/01_choix_probleme.ipynb), puis [la présentation du problème](../notebooks/02_presentation_probleme.ipynb) avec le kernel `.venv`. Exécuter les cellules du notebook 02 pour afficher les données et les contrôles. Poursuivre ensuite avec [l'analyse exploratoire](../notebooks/03_analyse_exploratoire.ipynb), dont la trame reste à compléter. Les notebooks de prétraitement et de modélisation existants restent réservés aux étapes suivantes.
4. Importer ultérieurement les modules avec `from src import preprocessing`, par exemple, sans modifier manuellement `sys.path`.
5. Construire les chemins avec `pathlib.Path`. Vérifier `Path.cwd()` et repérer explicitement la racine contenant `pyproject.toml`; le kernel peut démarrer à la racine ou dans `notebooks/`.
6. Après développement, redémarrer le kernel et exécuter toutes les cellules dans l'ordre.
7. Générer ultérieurement les CSV `Id,SalePrice` dans `submissions/`, dans l'ordre de `sample_submission.csv`; soumettre sur Kaggle et conserver captures, score et paramètres dans `docs/`.

## Reproductibilité scientifique

- Chaque étudiant utilise son propre `.venv` et Python 3.14.
- `pyproject.toml` centralise les dépendances, sans verrouillage exact. Après installation validée : `python -m pip freeze > docs/environnement-local.txt`. Vérifier et annoter toute ligne éditable contenant un chemin local avant partage; cet instantané documente l'expérience et ne remplace pas un verrouillage.
- Fixer et consigner un `random_state` pour les opérations aléatoires qui le permettent. Cela ne garantit pas une identité numérique entre plateformes.
- Séparer entraînement et validation avant les transformations apprises et les choix guidés par la cible.
- Ajuster imputation, encodage, standardisation et sélection dans les futurs pipelines, uniquement sur l'entraînement de chaque pli, sans apprendre de la validation ou de `test.csv`.
- Consigner chaque expérience : commit, versions, partition, graine, variables, transformations, modèle, paramètres, protocole, métriques et interprétation. Prévoir un tableau dans le notebook de modélisation et d'évaluation ou `docs/`.
- Préserver les données brutes et documenter provenance et date de téléchargement.
