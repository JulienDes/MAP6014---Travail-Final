"""Sélection par corrélation (étape 4). La qualité globale et la surface habitable sont fortement associées au prix."""


from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Sélectionner les caractéristiques selon leur corrélation avec la cible."""

    def fit(self, X, y=None):
        """Déterminer les colonnes à retenir à partir de X et des prix y d'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame limitée aux colonnes retenues."""
        raise NotImplementedError
