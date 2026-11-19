"""Transformation logarithmique, normalisation et standardisation (étape 4.5).

Trois notions souvent confondues :
- transformation logarithmique (log1p) : change la FORME d'une distribution à
  longue queue; log1p(x) = ln(1 + x) accepte les zéros;
- normalisation (min-max) : ramène chaque variable entre 0 et 1;
- standardisation : ramène chaque variable à une moyenne de 0 et un écart type de 1.

Ordre : d'abord la forme (log1p), ensuite l'échelle (standardisation).

Pourquoi une mise à l'échelle : Ridge pénalise la taille des coefficients, qui
dépend de l'unité. LotArea, en pi², exige un coefficient minuscule, presque pas
pénalisé; OverallQual, un gros coefficient, fortement pénalisé. Exprimer chaque
variable en écarts types rend la pénalité équitable. L'ACP cherche la plus grande
variance : sans mise à l'échelle, LotArea (variance ≈ 100 millions) dominerait
tout. Les modèles à base d'arbres n'en auraient pas besoin.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.steps.encoding import ORDINAL_COLUMNS

# Au-delà, la distribution a une longue queue à droite. Le seuil usuel 0,75 est
# trop proche de l'asymétrie des surfaces du sous-sol une fois les deux ventes
# aberrantes retirées (TotalBsmtSF ≈ 0,51) : leur log changeait d'un pli à
# l'autre. 0,5 les garde au log et réduit l'erreur (notebook, section 6.3).
SKEW_THRESHOLD = 0.5
# À partir de la moitié de zéros, l'asymétrie vient de la masse de zéros, pas
# d'une longue queue. Le log réduirait la variable à « présent ou absent » et
# effacerait les différences de taille (2ndFlrSF : 0 → 0, 316 → 5,76,
# 1 142 → 7,04). La présence est déjà captée par les indicateurs Has….
MAX_ZERO_SHARE = 0.5
# Une année, un mois, un score ou un niveau d'échelle n'est pas une grandeur à longue
# queue. KitchenAbvGr compte les cuisines (presque toujours 1) : son asymétrie vient
# de quelques maisons à 2 cuisines, pas d'une queue de grandes valeurs.
NO_LOG_COLUMNS = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold",
                  "OverallQual", "OverallCond", "KitchenAbvGr", *ORDINAL_COLUMNS]


class SkewCorrector(TransformerMixin, BaseEstimator):
    """Appliquer log1p aux colonnes à longue queue repérées sur l'entraînement.

    threshold=None désactive le log; max_zero_share=1.0 désactive l'exclusion
    des colonnes surtout nulles. Ces deux options servent aux comparaisons.
    """

    def __init__(self, threshold=SKEW_THRESHOLD, max_zero_share=MAX_ZERO_SHARE,
                 exclude=NO_LOG_COLUMNS):
        self.threshold = threshold
        self.max_zero_share = max_zero_share
        self.exclude = exclude

    def fit(self, X, y=None):
        """Choisir les colonnes à passer au log et documenter chaque décision."""
        rows = []
        for column in X.select_dtypes(include="number").columns:
            values = X[column]
            skew, zero_share = values.skew(), values.eq(0).mean()
            if column in self.exclude:
                reason = "Année, mois, score, comptage ou échelle ordinale"
            elif values.nunique() <= 2:
                reason = "Indicateur 0/1"
            elif values.lt(0).any():
                reason = "Valeurs négatives"
            elif self.threshold is None:
                reason = "Log désactivé"
            elif skew <= self.threshold:
                reason = "Asymétrie faible"
            elif zero_share >= self.max_zero_share:
                reason = "Surtout des zéros"
            else:
                reason = "Longue queue à droite"
            decision = "log1p" if reason == "Longue queue à droite" else "inchangée"
            rows.append([column, skew, zero_share, decision, reason])
        self.skew_report_ = pd.DataFrame(
            rows, columns=["Colonne", "Asymétrie", "Part de zéros", "Décision", "Raison"])
        self.log_columns_ = self.skew_report_.loc[
            self.skew_report_["Décision"].eq("log1p"), "Colonne"].tolist()
        return self

    def transform(self, X):
        """Renvoyer une copie où les colonnes retenues passent par log1p."""
        result = X.copy()
        result[self.log_columns_] = np.log1p(result[self.log_columns_])
        return result


def make_scaler(kind="standard"):
    """Construire la mise à l'échelle des colonnes numériques.

    "standard" donne StandardScaler (notre choix); "minmax" et "none" servent
    seulement à la comparaison du notebook. Les paramètres (moyennes et écarts
    types, ou minimums et maximums) sont appris dans fit, donc sur
    l'entraînement seulement.
    """
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}
    return scalers[kind]
