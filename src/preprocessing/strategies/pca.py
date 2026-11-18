"""ACP (étape 4). Quatre paires de caractéristiques ont une corrélation de Spearman d'au moins 0,8."""


from sklearn.base import BaseEstimator, TransformerMixin


class PCAReducer(TransformerMixin, BaseEstimator):
    """Réduire la dimension par analyse en composantes principales."""

    def fit(self, X, y=None):
        """Ajuster les axes de projection sur l'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer les projections dans un nouveau DataFrame."""
        raise NotImplementedError


class BlockPCA(TransformerMixin, BaseEstimator):
    """Réduire séparément des groupes de caractéristiques apparentées."""

    def fit(self, X, y=None):
        """Ajuster les axes de chaque bloc sur l'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer les projections par bloc dans un nouveau DataFrame."""
        raise NotImplementedError
