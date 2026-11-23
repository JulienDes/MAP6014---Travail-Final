"""Analyse en composantes principales (étape 4.6).

L'ACP remplace des colonnes corrélées par des combinaisons non corrélées entre
elles, les composantes, rangées de la plus grande variance à la plus petite.
Garder les premières réduit la dimension en perdant le moins de variance possible.
L'étape 3 (section 5) a relevé quatre paires de caractéristiques avec une
corrélation de Spearman d'au moins 0,8.

Les données doivent être standardisées (section 6) : sinon, la colonne de plus
grande variance, LotArea, formerait presque seule la première composante. L'ACP
ne regarde jamais le prix : une composante de faible variance peut pourtant
être utile à la prédiction. C'est sa limite principale.

PCAReducer projette toutes les colonnes reçues; BlockPCA projette séparément
des blocs de colonnes qui mesurent une même chose, ce qui garde des composantes
lisibles (« taille du garage ») et laisse les autres colonnes intactes.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

# Part de la variance des colonnes numériques à conserver; choisie par validation
# croisée (notebook, section 7.3).
PCA_VARIANCE = 0.99

# Après la sélection causale, ces colonnes décrivent un même équipement (présence,
# taille, qualité) et ont encore un Spearman d'au moins 0,8 (notebook, section 7.4).
# Deux paires aussi corrélées restent séparées : OverallQual et QualxArea, car
# l'interaction perdrait son rôle, et TotalBsmtSF et 1stFlrSF, deux étages dont le
# pi² n'a pas la même valeur (section 4.3).
BLOCKS = {
    "Pool": ["PoolArea", "PoolQC", "HasPool"],
    "Misc": ["MiscVal", "HasMiscFeature"],
    "Fireplace": ["Fireplaces", "FireplaceQu", "HasFireplace"],
    "Masonry": ["MasVnrArea", "HasMasonryVeneer"],
    "SecondFloor": ["2ndFlrSF", "Has2ndFloor"],
    "BsmtFin1": ["BsmtFinType1", "BsmtFinSF1"],
    "BsmtFin2": ["BsmtFinType2", "BsmtFinSF2"],
    "GarageState": ["GarageQual", "GarageCond"],
}


class PCAReducer(TransformerMixin, BaseEstimator):
    """Remplacer toutes les colonnes par leurs premières composantes principales.

    n_components entier : nombre de composantes; entre 0 et 1 : part de la
    variance à conserver.
    """

    def __init__(self, n_components=PCA_VARIANCE):
        self.n_components = n_components

    def fit(self, X, y=None):
        """Ajuster les axes de projection sur l'entraînement."""
        self.pca_ = PCA(n_components=self.n_components, svd_solver="full").fit(X)
        self.names_ = [f"PC{i + 1}" for i in range(self.pca_.n_components_)]
        return self

    def transform(self, X):
        """Renvoyer les projections dans un nouveau DataFrame."""
        return pd.DataFrame(self.pca_.transform(X), index=X.index, columns=self.names_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names_, dtype=object)


class BlockPCA(TransformerMixin, BaseEstimator):
    """Remplacer chaque bloc de colonnes apparentées par sa première composante.

    Les colonnes hors des blocs passent sans changement. Une colonne d'un bloc
    absente des données est ignorée.
    """

    def __init__(self, blocks=BLOCKS):
        self.blocks = blocks

    def fit(self, X, y=None):
        """Ajuster les axes de chaque bloc sur l'entraînement."""
        self.pcas_, self.block_columns_ = {}, {}
        for name, columns in self.blocks.items():
            present = [column for column in columns if column in X.columns]
            if len(present) >= 2:
                self.block_columns_[name] = present
                self.pcas_[name] = PCA(n_components=1).fit(X[present])
        used = {column for columns in self.block_columns_.values() for column in columns}
        self.passthrough_ = [column for column in X.columns if column not in used]
        return self

    def transform(self, X):
        """Renvoyer les colonnes hors blocs, puis une composante par bloc."""
        result = X[self.passthrough_].copy()
        for name, columns in self.block_columns_.items():
            result[f"{name}_PC1"] = self.pcas_[name].transform(X[columns])[:, 0]
        return result

    def loadings(self):
        """Poids de chaque colonne dans la composante de son bloc et variance expliquée."""
        rows = []
        for name, pca in self.pcas_.items():
            for column, weight in zip(self.block_columns_[name], pca.components_[0]):
                rows.append([name, column, weight, pca.explained_variance_ratio_[0]])
        return pd.DataFrame(rows, columns=["Bloc", "Colonne", "Poids", "Variance expliquée"])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.passthrough_, *(f"{name}_PC1" for name in self.block_columns_)],
                          dtype=object)
