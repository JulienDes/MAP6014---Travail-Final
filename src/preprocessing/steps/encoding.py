"""Encodage des variables catégorielles (étape 4.4).

Les modèles ne calculent qu'avec des nombres : chaque catégorie texte doit
devenir un ou plusieurs nombres. Il y a trois façons de le faire :

a) Numéros arbitraires (ordre alphabétique : a = 1, b = 2…) : à ÉVITER. Le
   modèle croit qu'une catégorie « vaut plus » qu'une autre. Par exemple,
   KitchenQual en ordre alphabétique donne Ex = 0, Fa = 1, Gd = 2, TA = 3 :
   une cuisine « moyenne » (TA) vaudrait plus qu'une cuisine « excellente » (Ex).
b) Encodage ordinal : des nombres qui suivent le VRAI ordre
   (Po = 1 < Fa = 2 < TA = 3 < Gd = 4 < Ex = 5). Il n'est correct que si la
   variable a un ordre réel de qualité ou de degré. Limite : il suppose des
   écarts égaux entre deux niveaux voisins.
c) Encodage one-hot : une colonne 0/1 par catégorie. Aucune catégorie ne
   reçoit un nombre plus grand qu'une autre, ce qui est nécessaire pour les
   variables sans ordre.

Règle retenue : ordinal pour les échelles réellement ordonnées, one-hot pour
tout le reste.

Ne pas confondre avec les indicateurs Has… de features.py : ils ont la même
forme (0/1), mais partent d'une MESURE numérique (surface > 0) et donnent une
colonne par équipement. Le one-hot part d'une variable TEXTE et donne une
colonne par catégorie.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# data_description.txt : Ex Excellent, Gd Good, TA Average/Typical, Fa Fair, Po Poor.
# Le NA « No Basement », « No Garage »… est devenu « None » à l'étape missing.py :
# l'absence se place sous le niveau le plus bas.
QUALITY_SCALE = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
# Les dix colonnes que data_description.txt décrit avec l'échelle Ex/Gd/TA/Fa/Po.
QUALITY_COLUMNS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                   "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
# data_description.txt : Gd Good Exposure > Av Average > Mn Mimimum > No No Exposure; NA No Basement.
BSMT_EXPOSURE_SCALE = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
# data_description.txt : GLQ Good Living Quarters > ALQ Average > BLQ Below Average
# > Rec Average Rec Room > LwQ Low Quality > Unf Unfinshed; NA No Basement.
BSMT_FINISH_SCALE = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
# data_description.txt : Fin Finished > RFn Rough Finished > Unf Unfinished; NA No Garage.
GARAGE_FINISH_SCALE = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
# data_description.txt : de Typ Typical Functionality à Sal Salvage only,
# en passant par des déductions mineures, modérées puis majeures.
FUNCTIONAL_SCALE = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5,
                    "Min1": 6, "Typ": 7}

# Les 15 échelles réellement ordonnées. Restent NOMINALES, malgré un ordre apparent :
# LotShape (irrégularité), LandSlope (pente), PavedDrive (pavage), CentralAir, Street
# et Alley décrivent un degré, pas clairement une valeur, et le one-hot n'impose
# aucune direction. Fence mélange deux dimensions (intimité et bois). MSSubClass est
# un code de type de logement, converti en texte au nettoyage.
ORDINAL_COLUMNS = {
    **{column: QUALITY_SCALE for column in QUALITY_COLUMNS},
    "BsmtExposure": BSMT_EXPOSURE_SCALE,
    "BsmtFinType1": BSMT_FINISH_SCALE,
    "BsmtFinType2": BSMT_FINISH_SCALE,
    "GarageFinish": GARAGE_FINISH_SCALE,
    "Functional": FUNCTIONAL_SCALE,
}
# Sous 10 maisons, une colonne 0/1 est presque vide; le test du notebook (section 5)
# ne montre aucune perte jusqu'à ce seuil.
RARE_CATEGORY_MIN_COUNT = 10


class QualityOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Remplacer les codes des échelles ordonnées par leur rang."""

    def __init__(self, scales=ORDINAL_COLUMNS):
        self.scales = scales

    def fit(self, X, y=None):
        """Renvoyer le transformateur : les échelles viennent du dictionnaire des données."""
        return self

    def transform(self, X):
        """Renvoyer une copie où chaque colonne ordonnée devient un entier."""
        result = X.copy()
        for column, scale in self.scales.items():
            # Après MissingValueHandler, un code inconnu signale une erreur en amont :
            # mieux vaut s'arrêter que de produire un rang inventé.
            unknown = sorted(set(result[column]) - set(scale))
            if unknown:
                raise ValueError(f"Valeurs hors échelle dans {column} : {', '.join(map(str, unknown))}")
            result[column] = result[column].map(scale).astype(int)
        return result


def nominal_columns(X):
    """Renvoyer les colonnes texte restantes, c'est-à-dire les variables nominales.

    À appliquer après QualityOrdinalEncoder, qui a déjà converti les échelles.
    """
    return X.select_dtypes(exclude="number").columns.tolist()


def make_onehot_encoder(min_count=RARE_CATEGORY_MIN_COUNT):
    """Construire l'encodeur one-hot des variables nominales.

    Les catégories et les regroupements sont appris dans fit, donc sur
    l'entraînement seulement. Une catégorie de moins de min_count maisons
    rejoint une colonne « autre » (infrequent) : une colonne presque vide ne
    permet pas d'estimer un effet fiable. Une catégorie absente de
    l'entraînement, comme MSSubClass 150 dans test.csv, va dans « autre » si
    cette colonne existe; sinon, toutes ses colonnes valent 0, sans erreur.
    Aucune colonne de référence n'est supprimée (drop) : la régularisation de
    Ridge gère la redondance et chaque colonne reste lisible.
    min_count=None désactive le regroupement.
    """
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_count,
                         sparse_output=False)
