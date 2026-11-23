"""Sélection par corrélation (étape 4.6).

Deux filtres, appris sur l'entraînement seulement :
1) pertinence : garder les colonnes dont la corrélation de Spearman avec le prix
   atteint min_target_corr en valeur absolue;
2) redondance : parcourir ces colonnes de la plus liée au prix à la moins liée,
   et écarter une colonne corrélée à au moins max_pair_corr avec une colonne déjà
   gardée. Sur deux colonnes redondantes, la plus liée au prix reste.

Spearman se calcule sur les rangs : log1p et la standardisation ne le changent
pas, et une relation monotone non linéaire compte entière. La sélection ignore
toute relation conjointe : une colonne faible seule, mais utile avec une autre,
est perdue. C'est la limite principale de cette stratégie.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Les deux seuils sont choisis par validation croisée (notebook, section 7.1).
# Lien minimal avec le prix : au-delà de 0,1, des colonnes utiles disparaissent.
MIN_TARGET_CORRELATION = 0.05
# Redondance : le seuil 0,8 de l'étape 3 retire aussi des colonnes proches mais
# distinctes, comme TotalBsmtSF et 1stFlrSF; 0,9 retire surtout de vrais doublons.
MAX_PAIR_CORRELATION = 0.9


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Sélectionner les colonnes liées au prix et non redondantes entre elles.

    max_pair_corr=None désactive le filtre de redondance (comparaison du notebook).
    """

    def __init__(self, min_target_corr=MIN_TARGET_CORRELATION, max_pair_corr=MAX_PAIR_CORRELATION):
        self.min_target_corr = min_target_corr
        self.max_pair_corr = max_pair_corr

    def fit(self, X, y):
        """Choisir les colonnes à partir de X et des prix y (ou de leur log) d'entraînement."""
        ranks = X.rank()
        # Une colonne constante dans le pli n'a pas de corrélation définie : lien nul.
        target = ranks.corrwith(pd.Series(np.asarray(y), index=X.index).rank()).abs().fillna(0)
        candidates = target[target.ge(self.min_target_corr)].sort_values(ascending=False).index
        pairs = ranks[candidates].corr().abs()

        kept, rows = [], []
        for column in target.sort_values(ascending=False).index:
            if column not in candidates:
                rows.append([column, target[column], "retirée", "Lien faible avec le prix", None])
                continue
            twins = pairs.loc[column, kept]
            redundant = (self.max_pair_corr is not None and len(kept) > 0
                         and twins.max() >= self.max_pair_corr)
            twin = twins.idxmax() if redundant else None
            if twin is None:
                kept.append(column)
                rows.append([column, target[column], "gardée", "Liée au prix, non redondante", None])
            else:
                rows.append([column, target[column], "retirée", "Redondante", twin])
        self.selected_columns_ = kept
        self.report_ = pd.DataFrame(rows, columns=["Colonne", "|Spearman| avec le prix", "Décision",
                                                   "Raison", "Colonne gardée à sa place"])
        return self

    def transform(self, X):
        """Renvoyer une copie du DataFrame limitée aux colonnes retenues."""
        return X[self.selected_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_columns_, dtype=object)
