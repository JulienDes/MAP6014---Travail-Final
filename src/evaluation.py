"""Évaluation (étape 6) sur l'échelle des dollars et celle du logarithme, qui réduit l'asymétrie des prix."""

import numpy as np

from src.preprocessing.steps.cleaning import unusual_sales


def rmse_log(y_true, y_pred):
    """Calculer la RMSE Kaggle sur les logarithmes naturels des prix positifs.

    root_mean_squared_log_error de scikit-learn utilise log1p, qui ajoute
    un dollar à chaque prix avant de prendre le logarithme.
    """
    errors = np.log(np.asarray(y_pred)) - np.log(np.asarray(y_true))
    return np.sqrt(np.mean(errors ** 2))


def cv_rmse(y_true, y_pred, X):
    """Calculer la RMSE log des prédictions hors pli sur les ventes typiques.

    Les ventes aberrantes (unusual_sales) sont retirées de l'entraînement de chaque
    pli; un modèle linéaire leur prédit alors un prix plusieurs fois trop élevé, et
    leurs deux carrés d'erreur domineraient la comparaison. y_true est en dollars,
    y_pred dans le même ordre que X et y_true.
    """
    typical = ~unusual_sales(X, y_true).to_numpy()
    return rmse_log(np.asarray(y_true)[typical], np.asarray(y_pred)[typical])


def regression_metrics(y_true, y_pred):
    """Calculer la MAE, la MSE, la RMSE et le R² en complément de la RMSE logarithmique."""
    raise NotImplementedError


def compare_results(results):
    """Réunir les métriques des modèles évalués sur les mêmes partitions."""
    raise NotImplementedError
