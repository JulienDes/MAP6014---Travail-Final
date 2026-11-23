"""Développement de la solution (étape 5). L'asymétrie de SalePrice motive la modélisation du logarithme du prix."""


def load_data(train_path, test_path):
    """Lire les CSV avec na_values=["NA"] et keep_default_na=False pour préserver la catégorie « None »."""
    raise NotImplementedError


def make_model():
    """Construire le modèle de régression du prix."""
    raise NotImplementedError


def holdout_evaluate(pipeline, X, y):
    """Ajuster le pipeline sur l'entraînement puis l'évaluer sur une partition réservée."""
    raise NotImplementedError


def cross_validate_model(pipeline, X, y, cv):
    """Évaluer le pipeline en le réajustant sur l'entraînement de chaque pli."""
    raise NotImplementedError
