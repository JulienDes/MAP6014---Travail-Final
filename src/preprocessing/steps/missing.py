"""Traiter quatre familles de valeurs absentes après le nettoyage.

Une absence d'équipement se code « None » ou zéro. Une fausse absence,
contredite par une mesure ou une description, se complète à partir des
maisons équipées. Une valeur sans objet, comme l'année d'un garage inexistant,
demande une convention. Une information réellement inconnue, comme la façade
du terrain, demande une estimation sur les maisons d'entraînement.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Une mesure positive atteste la présence. Ajouter une entrée suffit pour un nouvel équipement.
EQUIPMENT_GROUPS = {
    "Basement": {
        "measure": "TotalBsmtSF",
        "descriptors": ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
        "numbers": ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"],
    },
    "Garage": {
        "measure": "GarageArea",
        "descriptors": ["GarageType", "GarageFinish", "GarageQual", "GarageCond"],
        "numbers": ["GarageCars"],
    },
    "Fireplace": {"measure": "Fireplaces", "descriptors": ["FireplaceQu"], "numbers": []},
    "Pool": {"measure": "PoolArea", "descriptors": ["PoolQC"], "numbers": []},
    "MasonryVeneer": {"measure": "MasVnrArea", "descriptors": ["MasVnrType"], "numbers": []},
    "MiscFeature": {"measure": "MiscVal", "descriptors": ["MiscFeature"], "numbers": []},
}
# Le dictionnaire donne le sens de NA sans fournir de mesure de contrôle.
NO_MEASURE_ABSENCE = ["Alley", "Fence"]
# La catégorie déjà utilisée pour un parement absent convient aux autres équipements.
ABSENT_LABEL = "None"


def _equipment_cases(X, group):
    """Classer les maisons avant imputation pour le diagnostic et le traitement."""
    measure = X[group["measure"]]
    descriptions = X[group["descriptors"]]
    absent = descriptions.isna() | descriptions.isin([ABSENT_LABEL])
    incomplete = absent.any(axis=1) | X[group["numbers"]].isna().any(axis=1)
    equipped = measure.gt(0).fillna(False)
    return pd.DataFrame({
        "Complètes": equipped & ~incomplete,
        "R1": (measure.eq(0) | measure.isna()) & absent.all(axis=1),
        "R2": equipped & incomplete,
        "R3": measure.isna() & ~absent.all(axis=1),
    }, index=X.index)


def consistency_report(X, groups=EQUIPMENT_GROUPS):
    """Compter les cas par groupe avant imputation, puis les NA sans mesure associée."""
    rows = {name: _equipment_cases(X, group).sum().to_dict() for name, group in groups.items()}
    for column in NO_MEASURE_ABSENCE:
        if column in X:
            rows[column] = {"NA sans mesure": X[column].isna().sum()}
    return pd.DataFrame.from_dict(rows, orient="index").astype("Int64")


def _column_stat(series):
    """Calculer la médiane ou le premier mode disponible sur l'échantillon fourni."""
    known = series.dropna()
    if known.empty:
        return float("nan")
    if pd.api.types.is_numeric_dtype(series):
        return known.median()
    return known.mode().iat[0]


class MissingValueHandler(TransformerMixin, BaseEstimator):
    """Imputer par cohérence des équipements, puis par règles particulières et par colonne."""

    def __init__(self, groups=EQUIPMENT_GROUPS, garage_year_fill="year_built",
                 lot_frontage_fill="neighborhood"):
        self.groups = groups
        self.garage_year_fill = garage_year_fill
        self.lot_frontage_fill = lot_frontage_fill

    def fit(self, X, y=None):
        """Apprendre les statistiques uniquement sur les maisons d'entraînement nettoyées."""
        if self.garage_year_fill not in {"year_built", "zero", "drop"}:
            raise ValueError("Option garage_year_fill inconnue.")
        if self.lot_frontage_fill not in {"neighborhood", "global"}:
            raise ValueError("Option lot_frontage_fill inconnue.")
        self.column_fill_ = {col: _column_stat(X[col]) for col in X}
        self.equipment_fill_ = {}
        for group in self.groups.values():
            measure = group["measure"]
            equipped = X.loc[X[measure].gt(0)]
            for col in [measure, *group["numbers"]]:
                self.equipment_fill_[col] = _column_stat(equipped[col])
            for col in group["descriptors"]:
                self.equipment_fill_[col] = _column_stat(equipped[col].replace(ABSENT_LABEL, pd.NA))

        self.frontage_by_neighborhood_ = pd.Series(dtype=float)
        if {"LotFrontage", "Neighborhood"} <= set(X):
            self.frontage_by_neighborhood_ = X.groupby("Neighborhood")["LotFrontage"].median()
        return self

    def transform(self, X):
        """Imputer une copie et signaler les colonnes qui restent sans valeur de repli."""
        check_is_fitted(self, "column_fill_")
        result = X.copy()
        for group in self.groups.values():
            measure = group["measure"]
            cases = _equipment_cases(result, group)
            for col in group["descriptors"]:
                # Une description entièrement vide peut avoir été lue comme un flottant.
                if pd.api.types.is_numeric_dtype(result[col]) and result[col].isna().all():
                    result[col] = result[col].astype("string")
                result.loc[cases["R1"] & result[col].isna(), col] = ABSENT_LABEL
            for col in [measure, *group["numbers"]]:
                result.loc[cases["R1"] & result[col].isna(), col] = 0

            # R3 précède R2 pour reconnaître le garage de la maison 2577.
            # Un cabanon déclaré peut valoir zéro : cette mesure renseignée est légitime.
            for col in [measure, *group["numbers"]]:
                mask = cases["R3"] & result[col].isna()
                result[col] = result[col].mask(mask, self.equipment_fill_[col])
            equipped = result[measure].gt(0)
            for col in group["descriptors"]:
                mask = equipped & (result[col].isna() | result[col].eq(ABSENT_LABEL))
                result.loc[mask, col] = self.equipment_fill_[col]
            for col in group["numbers"]:
                mask = equipped & result[col].isna()
                result[col] = result[col].mask(mask, self.equipment_fill_[col])

        for col in NO_MEASURE_ABSENCE:
            if col in result:
                result[col] = result[col].fillna(ABSENT_LABEL)
        if "GarageYrBlt" in result:
            if self.garage_year_fill == "drop":
                result = result.drop(columns="GarageYrBlt")
            else:
                year = result["YearBuilt"] if self.garage_year_fill == "year_built" else 0
                result["GarageYrBlt"] = result["GarageYrBlt"].fillna(year)
        if self.lot_frontage_fill == "neighborhood" and {"LotFrontage", "Neighborhood"} <= set(result):
            local = result["Neighborhood"].map(self.frontage_by_neighborhood_)
            result["LotFrontage"] = result["LotFrontage"].fillna(local)
        if "Functional" in result:
            result["Functional"] = result["Functional"].fillna("Typ")

        # Le repli global couvre aussi un quartier inédit ou une colonne complète dans train.
        result = result.fillna(self.column_fill_)
        unresolved = result.columns[result.isna().any()].tolist()
        if unresolved:
            raise ValueError("Valeurs absentes après imputation dans : " + ", ".join(unresolved))
        return result
