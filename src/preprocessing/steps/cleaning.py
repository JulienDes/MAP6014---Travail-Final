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
# Sous ce prix, les très grandes maisons se détachent de la tendance (étape 3, § 7.1).
LOW_LARGE_HOUSE_PRICE = 300_000


class Cleaner(TransformerMixin, BaseEstimator):
    """Appliquer des règles fixes sans apprendre de paramètres sur les maisons."""

    def fit(self, X, y=None):
        """Renvoyer le transformateur, toutes les règles étant fixées à l'avance."""
        return self

    def transform(self, X):
        """Renvoyer un DataFrame nettoyé en conservant les lignes et leur index.

        Id identifie une vente et est retiré.
        MSSubClass devient un code textuel de type de logement.
        Exterior2nd reprend les codes de matériaux de Exterior1st.
        Utilities est écartée car sa seule exception connue est dans train.
        GarageYrBlt reprend YearRemodAdd si la date dépasse la tolérance après vente.
        MasVnrType et MasVnrArea sont réconciliés selon la présence et la surface.
        """
        cleaned = X.drop(columns="Id", errors="ignore").copy()
        cleaned["MSSubClass"] = cleaned["MSSubClass"].astype("string")
        cleaned["Exterior2nd"] = cleaned["Exterior2nd"].replace(EXTERIOR_CODES)
        cleaned = cleaned.drop(columns="Utilities")

        future = cleaned["GarageYrBlt"] > cleaned["YrSold"] + GARAGE_YEAR_TOLERANCE
        cleaned.loc[future, "GarageYrBlt"] = cleaned.loc[future, "YearRemodAdd"]

        area = cleaned["MasVnrArea"]
        absent = cleaned["MasVnrType"].eq("None")
        present = cleaned["MasVnrType"].isin(MASONRY_TYPES)
        cleaned.loc[absent & area.gt(0) & area.le(PLACEHOLDER_AREA), "MasVnrArea"] = 0
        cleaned.loc[absent & area.gt(PLACEHOLDER_AREA), "MasVnrType"] = np.nan
        cleaned.loc[present & area.eq(0), "MasVnrArea"] = np.nan
        return cleaned


def remove_outliers(X, y):
    """Retirer les ventes incohérentes de très grandes maisons à faible prix.

    Appliquer uniquement à l'entraînement de chaque pli, jamais à la validation
    ni à test.csv. La cible y est une série de prix en dollars indexée comme X.
    Renvoyer des copies des caractéristiques et des prix des maisons conservées.
    """
    unusual = X["GrLivArea"].gt(LARGE_HOUSE_AREA) & y.lt(LOW_LARGE_HOUSE_PRICE)
    return X.loc[~unusual].copy(), y.loc[~unusual].copy()
