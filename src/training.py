"""Développement de la solution (étape 5). L'asymétrie de SalePrice motive la modélisation du logarithme du prix.

Modèle retenu : la moyenne simple de cinq modèles de familles différentes, entraînés
sur la recette de prétraitement de l'étape 4 (make_base). La moyenne porte sur les
logarithmes des prix, l'échelle de la métrique Kaggle.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from src.evaluation import cv_rmse
from src.preprocessing.pipelines import make_base
from src.preprocessing.steps.cleaning import remove_outliers

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

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


def load_data(train_path, test_path):
    """Lire les CSV avec na_values=["NA"] et keep_default_na=False pour préserver la catégorie « None ».

    Renvoyer les caractéristiques et les prix de train, puis les caractéristiques de test.
    """
    options = {"na_values": ["NA"], "keep_default_na": False}
    train = pd.read_csv(train_path, **options)
    test = pd.read_csv(test_path, **options)
    return train.drop(columns="SalePrice"), train["SalePrice"], test


def make_model(members=None):
    """Construire le modèle de régression du prix.

    Prétraitement de l'étape 4, puis moyenne simple des membres (VotingRegressor).
    Le modèle apprend log(SalePrice) et prédit en dollars : la moyenne des
    logarithmes est une moyenne géométrique des prix. members (liste de noms de
    MEMBERS) permet d'entraîner un seul membre ou un sous-ensemble.
    """
    names = list(MEMBERS) if members is None else list(members)
    estimators = [(name, clone(MEMBERS[name])) for name in names]
    regressor = estimators[0][1] if len(names) == 1 else VotingRegressor(estimators)
    return TransformedTargetRegressor(make_pipeline(make_base(), regressor),
                                      func=np.log, inverse_func=np.exp)


def fit_model(model, X, y):
    """Ajuster le modèle sans les deux ventes aberrantes, retirées de l'entraînement seulement."""
    features, prices = remove_outliers(X, y)
    return model.fit(features, prices)


def member_predictions(model, X):
    """Prix prédits par chaque membre d'un ensemble ajusté, une colonne par modèle."""
    pipeline = model.regressor_
    transformed = pipeline[:-1].transform(X)
    voting = pipeline[-1]
    return pd.DataFrame({name: np.exp(estimator.predict(transformed))
                         for name, estimator in zip(voting.named_estimators_, voting.estimators_)},
                        index=X.index)


def holdout_evaluate(pipeline, X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE):
    """Ajuster le pipeline sur l'entraînement puis l'évaluer sur une partition réservée.

    Renvoyer le modèle ajusté, les prix prédits pour la partition réservée et leur
    RMSE log sur les ventes typiques.
    """
    train_index, valid_index = train_test_split(X.index, test_size=test_size,
                                                random_state=random_state)
    model = fit_model(clone(pipeline), X.loc[train_index], y.loc[train_index])
    predictions = pd.Series(model.predict(X.loc[valid_index]), index=valid_index)
    return model, predictions, cv_rmse(y.loc[valid_index], predictions, X.loc[valid_index])


def cross_validate_model(pipeline, X, y, cv):
    """Évaluer le pipeline en le réajustant sur l'entraînement de chaque pli.

    Renvoyer les prix prédits hors pli (une colonne par membre et une pour
    l'ensemble) et la RMSE log de chaque pli sur les ventes typiques.
    """
    columns = []
    scores = []
    for fit_index, valid_index in cv.split(X):
        model = fit_model(clone(pipeline), X.iloc[fit_index], y.iloc[fit_index])
        valid = X.iloc[valid_index]
        fold = (member_predictions(model, valid) if hasattr(model.regressor_[-1], "estimators_")
                else pd.DataFrame(index=valid.index))
        fold["Ensemble"] = model.predict(valid)
        columns.append(fold)
        scores.append(cv_rmse(y.iloc[valid_index], fold["Ensemble"], valid))
    return pd.concat(columns).loc[X.index], np.array(scores)
