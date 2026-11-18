"""Évaluation (étape 6) sur l'échelle des dollars et celle du logarithme, qui réduit l'asymétrie des prix."""

import numpy as np


def rmse_log(y_true, y_pred):
    """Calculer la RMSE Kaggle sur les logarithmes naturels des prix positifs.

    root_mean_squared_log_error de scikit-learn utilise log1p, qui ajoute
    un dollar à chaque prix avant de prendre le logarithme.
    """
    errors = np.log(np.asarray(y_pred)) - np.log(np.asarray(y_true))
    return np.sqrt(np.mean(errors ** 2))


def regression_metrics(y_true, y_pred):
    """Calculer la MAE, la MSE, la RMSE et le R² en complément de la RMSE logarithmique."""
    raise NotImplementedError


def compare_results(results):
    """Réunir les métriques des modèles évalués sur les mêmes partitions."""
    raise NotImplementedError
