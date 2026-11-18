"""Création de caractéristiques (étape 4.3). Les années brutes et les surfaces corrélées motivent des variables dérivées."""


from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(TransformerMixin, BaseEstimator):
    """Créer des caractéristiques telles que l'âge à la vente et la surface totale."""

    def fit(self, X, y=None):
        """Préparer les caractéristiques dérivées à partir des colonnes d'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame enrichie des caractéristiques dérivées."""
        raise NotImplementedError
