"""Sélection causale (étape 4). L'association entre âge et prix peut aussi refléter la taille ou la qualité des maisons."""


from sklearn.base import BaseEstimator, TransformerMixin


class CausalSelector(TransformerMixin, BaseEstimator):
    """Sélectionner les caractéristiques selon les hypothèses causales retenues."""

    def fit(self, X, y=None):
        """Déterminer les colonnes à retenir sur les données d'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame limitée aux colonnes retenues."""
        raise NotImplementedError
