"""Recettes de l'étape 4 pour comparer le traitement commun et les stratégies de réduction des redondances."""


def make_base():
    """Construire le pipeline avec les cinq étapes communes."""
    raise NotImplementedError


def make_correlation():
    """Construire le pipeline avec sélection par corrélation."""
    raise NotImplementedError


def make_causal():
    """Construire le pipeline avec sélection causale."""
    raise NotImplementedError


def make_pca():
    """Construire le pipeline avec analyse en composantes principales."""
    raise NotImplementedError


def make_causal_pca():
    """Construire le pipeline combinant sélection causale et ACP par blocs."""
    raise NotImplementedError
