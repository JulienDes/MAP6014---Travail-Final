"""Nettoyage des maisons d’Ames (étape 4.1).

Corriger les erreurs manifestes, harmoniser les codes et retirer les colonnes
sans information prédictive. Lorsque la correction est incertaine, marquer
la valeur inconnue (NaN) sans la deviner. L'étape des valeurs manquantes
(missing.py) apprendra les valeurs d'imputation sur l'entraînement.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Les deux colonnes de revêtement doivent désigner un même matériau par le même code.
EXTERIOR_CODES = {"Brk Cmn": "BrkComm", "CmentBd": "CemntBd", "Wd Shng": "WdShing"}
# Une vente sur plan peut précéder la fin de la construction d'une année.
GARAGE_YEAR_TOLERANCE = 1
# Un pied carré est une surface bouche-trou lorsque le parement est déclaré absent.
PLACEHOLDER_AREA = 1
# Ces codes du dictionnaire attestent la présence d'un parement.
MASONRY_TYPES = {"BrkFace", "BrkCmn", "Stone"}
# De Cock (2011) signale les surfaces habitables dépassant cette limite.
LARGE_HOUSE_AREA = 4_000
# Sous ce prix, les très grandes maisons se détachent de la tendance (notebook 03, section 6).
LOW_LARGE_HOUSE_PRICE = 300_000


class Cleaner(TransformerMixin, BaseEstimator):
    """Appliquer des règles fixes sans apprendre de paramètres sur les maisons."""

    def fit(self, X, y=None):
        """Renvoyer le transformateur, toutes les règles étant fixées à l'avance.

        Un Pipeline appelle fit sur chaque étape, qui doit renvoyer self
        (guide des estimateurs, section « Fitting ») :
        https://scikit-learn.org/stable/developers/develop.html#:~:text=even%20unsupervised%20estimators%20need%20to%20accept
        Scikit-learn garde aussi un fit inutile dans Normalizer, une étape sans état :
        https://scikit-learn.org/stable/modules/preprocessing.html#:~:text=the%20class%20is%20stateless
        """
        return self

    def transform(self, X):
        """Renvoyer un DataFrame nettoyé en conservant les lignes et leur index.

        Id identifie une vente et est retiré.
        MSSubClass est convertie en texte pour être traitée comme une variable catégorielle.
        Exterior2nd corrige trois codes pour les écrire comme dans Exterior1st.
        Utilities est écartée car sa seule exception connue est dans train.
        YearRemodAdd reprend YearBuilt si la rénovation précède la construction.
        GarageYrBlt reprend YearRemodAdd si la date dépasse la tolérance après vente.
        MasVnrType et MasVnrArea sont réconciliés selon la présence et la surface.
        """
        cleaned = X.drop(columns="Id", errors="ignore").copy()
        cleaned["MSSubClass"] = cleaned["MSSubClass"].astype("string")
        cleaned["Exterior2nd"] = cleaned["Exterior2nd"].replace(EXTERIOR_CODES)
        cleaned = cleaned.drop(columns="Utilities")

        # Sans rénovation, le dictionnaire donne YearRemodAdd = YearBuilt.
        early = cleaned["YearRemodAdd"] < cleaned["YearBuilt"]
        cleaned.loc[early, "YearRemodAdd"] = cleaned.loc[early, "YearBuilt"]

        future = cleaned["GarageYrBlt"] > cleaned["YrSold"] + GARAGE_YEAR_TOLERANCE
        cleaned.loc[future, "GarageYrBlt"] = cleaned.loc[future, "YearRemodAdd"]

        area = cleaned["MasVnrArea"]
        absent = cleaned["MasVnrType"].eq("None")
        present = cleaned["MasVnrType"].isin(MASONRY_TYPES)
        cleaned.loc[absent & area.gt(0) & area.le(PLACEHOLDER_AREA), "MasVnrArea"] = 0
        cleaned.loc[absent & area.gt(PLACEHOLDER_AREA), "MasVnrType"] = np.nan
        cleaned.loc[present & area.eq(0), "MasVnrArea"] = np.nan
        return cleaned


def unusual_sales(X, y):
    """Marquer les ventes incohérentes de très grandes maisons à faible prix (524 et 1299).

    Renvoyer un masque booléen indexé comme X. La cible y est en dollars.
    """
    return X["GrLivArea"].gt(LARGE_HOUSE_AREA) & y.lt(LOW_LARGE_HOUSE_PRICE)


def remove_outliers(X, y):
    """Retirer les ventes incohérentes de très grandes maisons à faible prix."""
    unusual = unusual_sales(X, y)
    return X.loc[~unusual].copy(), y.loc[~unusual].copy()
