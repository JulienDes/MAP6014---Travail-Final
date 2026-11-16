# Installation et utilisation

[Retour au README](../README.md)

### Prérequis

Windows, PowerShell, Git, VS Code avec les extensions **Python** et **Jupyter**, et CPython **3.14 standard, 64 bits**.

Python 3.14 est compatible avec les versions récentes retenues, notamment [NumPy 2.3.3](https://numpy.org/doc/stable/release/2.3.3-notes.html), [pandas 2.3.3](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v2.3.3.html), [SciPy 1.16.1](https://discuss.scientific-python.org/t/ann-scipy-1-16-1-release/2033), [scikit-learn](https://github.com/scikit-learn/scikit-learn/releases), [Matplotlib 3.10.7](https://pypi.org/project/matplotlib/3.10.7/) et [ipykernel 7.0.1](https://github.com/ipython/ipykernel/releases). Les bornes évitent certaines anciennes versions incompatibles; elles ne verrouillent pas exactement l'environnement.

`.python-version` indique 3.14 sans installer ni activer Python. `pyproject.toml` limite le projet à cette version mineure.

### 1. Vérifier Python et Git

```powershell
py --list
py -3.14 --version
git --version
```

Si nécessaire, installer Python 3.14 depuis python.org puis rouvrir PowerShell. Si `py` est absent, remplacer `py -3.14` par le chemin complet de l'exécutable Python 3.14 installé.

### 2. Cloner le dépôt privé et ouvrir VS Code

Remplacer l'URL d'exemple par celle du dépôt auquel le collègue a été invité; s'authentifier auprès de GitHub.

```powershell
git clone https://github.com/VOTRE-ORGANISATION/VOTRE-DEPOT.git house-prices
cd house-prices
code .
```

Si le dépôt est déjà présent, ouvrir directement sa racine. Si `code` est indisponible, utiliser **Fichier > Ouvrir le dossier**. Le nom `house-prices` est indicatif; le dossier actuel peut porter un autre nom.

### 3. Créer et activer `.venv`

Depuis la racine, dans le terminal PowerShell :

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
```

Le chemin affiché doit désigner `.venv\Scripts\python.exe`. Si PowerShell bloque l'activation, autoriser les scripts pour cette session :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Cette modification cesse à la fermeture du terminal. Si une politique institutionnelle la bloque, utiliser directement l'exécutable sans activation, par exemple : `& .\.venv\Scripts\python.exe -m pip install -e .`.

### 4. Mettre à jour pip et installer

Avec `.venv` activé :

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
python -c "import src.preprocessing, src.training, src.evaluation; import pandas, numpy, matplotlib, seaborn, scipy, sklearn, ipykernel; print('Imports OK')"
```

La distribution s'appelle `map6014-house-prices`; le paquet importé s'appelle `src`. La déclaration explicite `packages = ["src"]` permet l'installation avec l'arborescence demandée. L'installation éditable reflète les modifications sans réinstallation; redémarrer le kernel après modification si nécessaire. Ne pas installer un autre paquet nommé `src` dans cet environnement.

### 5. Sélectionner l'interpréteur et le kernel Jupyter

1. Ouvrir la palette VS Code avec `Ctrl+Shift+P`.
2. Choisir **Python: Select Interpreter**, puis `.venv\Scripts\python.exe`.
3. Ouvrir un notebook et cliquer **Select Kernel** en haut à droite.
4. Choisir **Python Environments**, puis le Python de `.venv`.
5. Si nécessaire, sélectionner manuellement le chemin de l'interpréteur.

`ipykernel` est inclus dans les dépendances. Un enregistrement global du kernel et un serveur Jupyter séparé ne sont normalement pas nécessaires dans VS Code. Si `.venv` n'apparaît pas, recharger la fenêtre et vérifier l'interpréteur. Le kernel et l'activation du terminal sont deux réglages distincts.

### 6. Désactiver et réactiver

```powershell
deactivate
# Dans un nouveau terminal, revenir à la racine, puis :
.\.venv\Scripts\Activate.ps1
```

Ne pas recréer `.venv` à chaque séance. Réexécuter `python -m pip install -e .` si les dépendances changent.

## Utilisation

1. Consulter [Data sur Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), vérifier les règles et télécharger manuellement les fichiers autorisés.
2. Décompresser directement dans `data/` : `train.csv`, `test.csv`, `sample_submission.csv` et `data_description.txt`.
3. Ouvrir les trois notebooks dans l'ordre avec le kernel `.venv`. Ils ne contiennent actuellement que des sections Markdown à compléter.
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
- Consigner chaque expérience : commit, versions, partition, graine, variables, transformations, modèle, paramètres, protocole, métriques et interprétation. Prévoir un tableau dans le troisième notebook ou `docs/`.
- Préserver les données brutes et documenter provenance et date de téléchargement.
