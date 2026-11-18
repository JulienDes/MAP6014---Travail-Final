"""Nettoyage (étape 4.1) des codes de revêtement et des années incohérentes, dont GarageYrBlt = 2207."""


from sklearn.base import BaseEstimator, TransformerMixin


class Cleaner(TransformerMixin, BaseEstimator):
    """Harmoniser les codes et corriger les incohérences entre caractéristiques."""

    def fit(self, X, y=None):
        """Préparer le nettoyage à partir des maisons d'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame avec les codes et les valeurs corrigés."""
        raise NotImplementedError


def remove_outliers(X, y):
    """Renvoyer X et y alignés après exclusion des ventes 524 et 1299 de l'entraînement."""
    raise NotImplementedError
