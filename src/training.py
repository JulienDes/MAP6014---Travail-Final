"""Développement de la solution (étape 5). L'asymétrie de SalePrice motive la modélisation du logarithme du prix.

Modèle retenu : la moyenne simple de cinq modèles de familles différentes, sur les
logarithmes des prix, l'échelle de la métrique Kaggle. Chaque membre a son propre
pipeline, avec le prétraitement de l'étape 4 adapté à sa famille (make_preprocessing).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from src.evaluation import cv_rmse
from src.preprocessing.pipelines import make_base
from src.preprocessing.steps.cleaning import remove_outliers

RANDOM_STATE = 42

# LassoCV et ElasticNetCV essaient de très faibles pénalités, lentes à converger et
# jamais retenues : leurs avertissements sont masqués.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PENALTIES = np.logspace(-4.5, -1.5, 40)
MEMBERS = {
    # Linéaires : le log du prix est presque une somme d'effets (étape 4).
    "Lasso": LassoCV(alphas=PENALTIES, max_iter=50_000),
    "ElasticNet": ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], alphas=PENALTIES,
                               max_iter=50_000),
    # Noyau polynomial de degré 2 : ajoute implicitement carrés et produits de colonnes.
    "Kernel Ridge": KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5),
    # Arbres : seuils et interactions; la perte de Huber atténue les prix extrêmes.
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=4, max_features="sqrt",
        min_samples_leaf=15, min_samples_split=10, loss="huber", random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=4, min_child_weight=2, subsample=0.7,
        colsample_bytree=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1,
        tree_method="hist"),
}

# Les arbres n'ont besoin ni du log ni de la mise à l'échelle (étape 4, section 8.1).
TREE_MEMBERS = {"Gradient Boosting", "XGBoost"}


def load_data(train_path, test_path):
    """Lire les CSV avec na_values=["NA"] et keep_default_na=False pour préserver la catégorie « None ».

    Renvoyer les caractéristiques et les prix de train, puis les caractéristiques de test.
    """
    options = {"na_values": ["NA"], "keep_default_na": False}
    train = pd.read_csv(train_path, **options)
    test = pd.read_csv(test_path, **options)
    return train.drop(columns="SalePrice"), train["SalePrice"], test


def make_preprocessing(name):
    """Prétraitement de l'étape 4 adapté à la famille du modèle name."""
    return make_base(log=False, scaler="none") if name in TREE_MEMBERS else make_base()


def make_model(members=None):
    """Construire le modèle de régression du prix.

    Chaque membre devient un pipeline : make_preprocessing(name), puis le modèle.
    Avec plusieurs membres, VotingRegressor fait la moyenne simple de ces pipelines.
    Le modèle apprend log(SalePrice) et prédit en dollars : la moyenne des
    logarithmes est une moyenne géométrique des prix. members (liste de noms de
    MEMBERS) permet d'entraîner un seul membre ou un sous-ensemble.
    """
    names = list(MEMBERS) if members is None else list(members)
    pipelines = [(name, make_pipeline(make_preprocessing(name), clone(MEMBERS[name])))
                 for name in names]
    regressor = pipelines[0][1] if len(names) == 1 else VotingRegressor(pipelines)
    return TransformedTargetRegressor(regressor, func=np.log, inverse_func=np.exp)


def fit_model(model, X, y):
    """Ajuster le modèle sans les deux ventes aberrantes, retirées de l'entraînement seulement."""
    features, prices = remove_outliers(X, y)
    return model.fit(features, prices)


def member_predictions(model, X):
    """Prix prédits par chaque membre d'un ensemble ajusté, une colonne par modèle.

    Chaque pipeline membre prétraite X lui-même.
    """
    voting = model.regressor_
    return pd.DataFrame({name: np.exp(pipeline.predict(X))
                         for name, pipeline in zip(voting.named_estimators_, voting.estimators_)},
                        index=X.index)


def cross_validate_model(pipeline, X, y, cv):
    """Évaluer le modèle en le réajustant sur l'entraînement de chaque pli.

    Renvoyer les prix prédits hors pli (une colonne par membre, pour un ensemble,
    et une pour l'ensemble) et la RMSE log de chaque pli sur les ventes typiques.
    """
    columns = []
    scores = []
    for fit_index, valid_index in cv.split(X):
        model = fit_model(clone(pipeline), X.iloc[fit_index], y.iloc[fit_index])
        valid = X.iloc[valid_index]
        fold = (member_predictions(model, valid) if isinstance(model.regressor_, VotingRegressor)
                else pd.DataFrame(index=valid.index))
        fold["Ensemble"] = model.predict(valid)
        columns.append(fold)
        scores.append(cv_rmse(y.iloc[valid_index], fold["Ensemble"], valid))
    return pd.concat(columns).loc[X.index], np.array(scores)
