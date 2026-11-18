"""Évaluation (étape 6) sur l'échelle des dollars et celle du logarithme, qui réduit l'asymétrie des prix."""


def rmse_log(y_true, y_pred):
    """Calculer la racine de l'erreur quadratique moyenne entre logarithmes naturels des prix."""
    raise NotImplementedError


def regression_metrics(y_true, y_pred):
    """Calculer la MAE, la MSE, la RMSE et le R² en complément de la RMSE logarithmique."""
    raise NotImplementedError


def compare_results(results):
    """Réunir les métriques des modèles évalués sur les mêmes partitions."""
    raise NotImplementedError
