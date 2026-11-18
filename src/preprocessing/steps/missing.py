"""Valeurs manquantes (étape 4.2). Un garage absent et une façade de terrain inconnue appellent des traitements distincts."""


from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(TransformerMixin, BaseEstimator):
    """Distinguer l'absence d'équipement des valeurs inconnues."""

    def fit(self, X, y=None):
        """Estimer les valeurs d'imputation sur les maisons d'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame avec les absences et les inconnues traitées."""
        raise NotImplementedError
