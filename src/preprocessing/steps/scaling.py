"""Normalisation et standardisation (étape 4.5). Parmi 35 variables numériques, 18 ont une asymétrie supérieure à 1."""


from sklearn.base import BaseEstimator, TransformerMixin


class SkewCorrector(TransformerMixin, BaseEstimator):
    """Appliquer log1p aux caractéristiques asymétriques compatibles."""

    def fit(self, X, y=None):
        """Identifier sur l'entraînement les colonnes à transformer."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame après transformation des colonnes retenues."""
        raise NotImplementedError


def make_scaler():
    """Construire un standardiseur à ajuster sur les données d'entraînement."""
    raise NotImplementedError
