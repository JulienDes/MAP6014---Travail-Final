"""Sélection causale (étape 4.6).

La sélection par corrélation choisit d'après les données; celle-ci choisit d'après
un raisonnement sur ce qui fait le prix, fixé avant de regarder les scores. Le prix
dépend de quelques causes : la taille, la qualité, l'âge, l'emplacement, les
équipements et les conditions de la vente. Chaque cause doit être mesurée une seule
fois. Trois règles retirent les colonnes qui la mesurent une deuxième fois :

1) un doublon porte la même information qu'une colonne gardée (YearBuilt et AgeAtSale);
2) une conséquence découle d'une cause gardée : le nombre de pièces découle de la
   surface habitable, le nombre de places du garage découle de sa surface. Elle
   n'ajoute que du bruit à la cause, au risque de partager son effet avec elle;
3) chaque surface est comptée une fois : jamais une somme avec toutes ses parties.
   Le mode « totals » garde les sommes, le mode « parts » garde les parties.

Aucune colonne n'est retirée parce que son lien avec le prix est faible : une
cause réelle, comme une piscine, reste même si elle est rare. Les causes communes
restent aussi ensemble : l'âge reste avec la taille et la qualité. L'effet
apparent de l'âge (étape 3) reflète en partie ces deux causes; sans elles, son
coefficient les absorberait.

Aucun apprentissage : les règles sont fixées à l'avance, identiques pour train et test.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Mode retenu : la validation croisée préfère les parties (notebook, section 7.2),
# car un pi² de sous-sol, de rez-de-chaussée et d'étage n'a pas forcément la même valeur.
CAUSAL_MODE = "parts"

# Règles 1 et 2, valables dans les deux modes.
DUPLICATES = {
    "YearBuilt": "Doublon de AgeAtSale (même information, au signe près)",
    "YearRemodAdd": "Doublon de YearsSinceRemodel (même information, au signe près)",
    "GarageYrBlt": "Doublon de l’âge : égale YearBuilt pour 79 % des garages et pour les maisons sans garage",
}
CONSEQUENCES = {
    "TotRmsAbvGrd": "Conséquence de GrLivArea : la surface habitable détermine le nombre de pièces",
    "GarageCars": "Conséquence de GarageArea : la surface du garage détermine le nombre de places",
}
# Règle 3. GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF et
# TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF dans les données d'origine.
SURFACE_RULES = {
    "totals": {
        **{column: "Partie de TotalSF ou de GrLivArea" for column in ("1stFlrSF", "2ndFlrSF")},
        "TotalBsmtSF": "Égale TotalSF − GrLivArea, à LowQualFinSF près",
        "BsmtUnfSF": "Reste du sous-sol : TotalSF − GrLivArea − BsmtFinSF1 − BsmtFinSF2",
        **{column: "Partie de TotalBath" for column in ("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath")},
        **{column: "Partie de TotalPorchSF"
           for column in ("WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch")},
    },
    "parts": {
        "TotalSF": "Somme de TotalBsmtSF, 1stFlrSF et 2ndFlrSF",
        "GrLivArea": "Somme de 1stFlrSF, 2ndFlrSF et LowQualFinSF",
        "BsmtUnfSF": "Reste du sous-sol : TotalBsmtSF − BsmtFinSF1 − BsmtFinSF2",
        "TotalBath": "Somme des quatre colonnes de salles de bain",
        "TotalPorchSF": "Somme des cinq colonnes de galeries",
    },
}


def causal_drops(mode="totals"):
    """Renvoyer les colonnes retirées et la raison de chaque retrait."""
    return {**DUPLICATES, **CONSEQUENCES, **SURFACE_RULES[mode]}


class CausalSelector(TransformerMixin, BaseEstimator):
    """Retirer les colonnes qui mesurent une deuxième fois une cause déjà présente.

    À placer après FeatureCreator, qui crée les sommes et les âges.
    """

    def __init__(self, mode=CAUSAL_MODE):
        self.mode = mode

    def fit(self, X, y=None):
        """Vérifier que les colonnes à retirer existent; aucune statistique n'est apprise."""
        drops = causal_drops(self.mode)
        absent = sorted(set(drops) - set(X.columns))
        if absent:
            raise ValueError("Colonnes attendues absentes : " + ", ".join(absent)
                             + ". Exécuter FeatureCreator avant CausalSelector.")
        self.dropped_columns_ = list(drops)
        self.selected_columns_ = [column for column in X.columns if column not in drops]
        return self

    def transform(self, X):
        """Renvoyer une copie du DataFrame limitée aux colonnes retenues."""
        return X[self.selected_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_columns_, dtype=object)
