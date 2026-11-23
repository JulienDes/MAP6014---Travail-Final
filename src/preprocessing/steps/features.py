"""Création de caractéristiques (étape 4.3).

Cette étape AJOUTE des variables calculées à partir des colonnes existantes et
n'en retire aucune. La création propose des candidates; la sélection
(strategies/) décidera plus tard lesquelles garder, par exemple TotalSF ou ses
trois parties, AgeAtSale ou YearBuilt.

Une variable créée doit respecter cinq critères :
1) être calculable avec nos colonnes, sans SalePrice. Sinon, il y a fuite : le
   prix au pi² est interdit. Une orientation nord/sud est impossible, car elle
   n'est pas dans les données;
2) avoir un sens pour un évaluateur immobilier, pas de combinaison au hasard;
3) apporter quelque chose de nouveau : un calcul que le modèle linéaire ne peut
   pas faire seul (produit, seuil) OU un résumé de plusieurs colonnes en une seule;
4) ne pas répéter une autre création;
5) avoir un lien réel avec le prix, meilleur que celui de ses parties.

Aucun apprentissage : fit renvoie self et les calculs sont identiques pour
train et test. Les données doivent déjà être passées par Cleaner, puis par
MissingValueHandler.
"""

from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.steps.missing import EQUIPMENT_GROUPS

# Une salle d'eau (toilette et lavabo) compte pour une demi-salle de bain,
# comme dans l'usage immobilier nord-américain.
SUM_FEATURES = {
    "TotalSF": {"TotalBsmtSF": 1, "1stFlrSF": 1, "2ndFlrSF": 1},
    "TotalBath": {"FullBath": 1, "HalfBath": 0.5, "BsmtFullBath": 1, "BsmtHalfBath": 0.5},
    "TotalPorchSF": {"WoodDeckSF": 1, "OpenPorchSF": 1, "EnclosedPorch": 1,
                     "3SsnPorch": 1, "ScreenPorch": 1},
}
# L'âge au moment de la vente se lit mieux qu'une année brute. Il est ramené à 0
# si négatif : une maison de test est vendue sur plan en 2007 et construite en 2008.
AGE_FEATURES = {"AgeAtSale": "YearBuilt", "YearsSinceRemodel": "YearRemodAdd"}
# L'étage n'est pas un équipement de EQUIPMENT_GROUPS, mais sa présence change le type de maison.
EXTRA_INDICATORS = {"Has2ndFloor": "2ndFlrSF"}
# Une maison vendue l'année de sa construction ou l'année suivante est neuve,
# ce qui couvre aussi les ventes sur plan.
NEW_HOUSE_MAX_AGE = 1


class FeatureCreator(TransformerMixin, BaseEstimator):
    """Ajouter 14 caractéristiques dérivées sans retirer de colonne."""

    def __init__(self, groups=EQUIPMENT_GROUPS):
        self.groups = groups

    def fit(self, X, y=None):
        """Renvoyer le transformateur : toutes les formules sont fixées à l'avance.

        Un Pipeline appelle fit sur chaque étape, qui doit renvoyer self
        (guide des estimateurs, section « Fitting ») :
        https://scikit-learn.org/stable/developers/develop.html#:~:text=even%20unsupervised%20estimators%20need%20to%20accept
        Scikit-learn garde aussi un fit inutile dans Normalizer, une étape sans état :
        https://scikit-learn.org/stable/modules/preprocessing.html#:~:text=the%20class%20is%20stateless
        """
        return self

    def transform(self, X):
        """Renvoyer une copie de X enrichie des sommes, âges, indicateurs et interaction."""
        # Ajouter un équipement à EQUIPMENT_GROUPS crée son indicateur sans autre code.
        indicators = {f"Has{name}": group["measure"] for name, group in self.groups.items()}
        indicators.update(EXTRA_INDICATORS)

        used = [col for parts in SUM_FEATURES.values() for col in parts]
        used += [*AGE_FEATURES.values(), "YrSold", *indicators.values(), "OverallQual", "GrLivArea"]
        missing = [col for col in dict.fromkeys(used) if X[col].isna().any()]
        if missing:
            raise ValueError("Valeurs absentes dans " + ", ".join(missing)
                             + " : exécuter MissingValueHandler avant FeatureCreator.")

        result = X.copy()
        for name, weights in SUM_FEATURES.items():
            result[name] = sum(weight * X[col] for col, weight in weights.items())
        for name, column in AGE_FEATURES.items():
            result[name] = (X["YrSold"] - X[column]).clip(lower=0)
        # HasGarage est aussi indispensable parce que GarageYrBlt d'une maison sans
        # garage a reçu YearBuilt à l'étape précédente : lui seul signale l'absence.
        for name, measure in indicators.items():
            result[name] = X[measure].gt(0).astype(int)
        result["IsNewHouse"] = X["YrSold"].le(X["YearBuilt"] + NEW_HOUSE_MAX_AGE).astype(int)
        # Un pi² de grande qualité vaut plus : seul calcul ici qu'un modèle linéaire
        # ne peut pas construire lui-même à partir des colonnes d'origine.
        result["QualxArea"] = X["OverallQual"] * X["GrLivArea"]
        return result
