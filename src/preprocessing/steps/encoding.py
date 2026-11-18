"""Encodage (étape 4.4) des qualités ordonnées et des catégories nominales, parfois inédites en validation."""


from sklearn.base import BaseEstimator, TransformerMixin


class QualityOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Encoder les appréciations de qualité en respectant leur ordre."""

    def fit(self, X, y=None):
        """Préparer les correspondances ordinales pour les colonnes d'entraînement."""
        raise NotImplementedError

    def transform(self, X):
        """Renvoyer une copie du DataFrame où les qualités sont encodées."""
        raise NotImplementedError


def make_onehot_encoder():
    """Construire un encodeur disjonctif tolérant aux catégories inconnues."""
    raise NotImplementedError
