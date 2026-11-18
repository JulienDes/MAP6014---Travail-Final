"""Vérifier les règles de nettoyage et leurs effets sur les maisons d'Ames."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import rmse_log
from src.preprocessing.steps.cleaning import Cleaner, remove_outliers


@pytest.fixture
def house():
    return pd.DataFrame([{
        "Id": 12, "MSSubClass": 60, "Exterior2nd": "Wd Shng", "Utilities": "AllPub",
        "YrSold": 2007, "YearRemodAdd": 2006, "GarageYrBlt": 2007,
        "MasVnrType": "None", "MasVnrArea": 0.0,
    }], index=[11])


def test_drop_optional_id(house):
    result = Cleaner().transform(house)
    assert "Id" not in result
    pd.testing.assert_frame_equal(result, Cleaner().transform(house.drop(columns="Id")))


def test_housing_type_is_text(house):
    assert Cleaner().transform(house).at[11, "MSSubClass"] == "60"


@pytest.mark.parametrize("original, expected", [
    ("Brk Cmn", "BrkComm"), ("CmentBd", "CemntBd"), ("Wd Shng", "WdShing"), ("VinylSd", "VinylSd"),
])
def test_exterior_codes(house, original, expected):
    assert Cleaner().transform(house.assign(Exterior2nd=original)).at[11, "Exterior2nd"] == expected


def test_drop_only_utilities_and_id(house):
    for column in ("Street", "PoolQC", "Condition2", "RoofMatl", "Heating", "MiscFeature"):
        house[column] = "rare"
    result = Cleaner().transform(house)
    assert list(result) == list(house.drop(columns=["Id", "Utilities"]))


@pytest.mark.parametrize("year, expected", [(2207, 2006), (2008, 2008)])
def test_garage_tolerance(house, year, expected):
    assert Cleaner().transform(house.assign(GarageYrBlt=year)).at[11, "GarageYrBlt"] == expected


@pytest.mark.parametrize("kind, area, expected_kind, expected_area", [
    ("None", 0.5, "None", 0), ("None", 1, "None", 0), ("None", 285, np.nan, 285),
    ("BrkFace", 0, "BrkFace", np.nan), ("BrkCmn", 0, "BrkCmn", np.nan),
    ("Stone", 0, "Stone", np.nan), ("Stone", 1, "Stone", 1),
    ("None", np.nan, "None", np.nan), (np.nan, 1, np.nan, 1),
])
def test_masonry(house, kind, area, expected_kind, expected_area):
    house["MasVnrType"] = pd.Series([kind], index=house.index, dtype="string")
    result = Cleaner().transform(house.assign(MasVnrArea=area))
    np.testing.assert_equal(result.at[11, "MasVnrArea"], expected_area)
    if pd.isna(expected_kind):
        assert pd.isna(result.at[11, "MasVnrType"])
    else:
        assert result.at[11, "MasVnrType"] == expected_kind


def test_cleaner_preserves_input(house):
    original = house.copy()
    cleaner = Cleaner()
    assert cleaner.fit(house) is cleaner
    result = cleaner.transform(house)
    result.at[11, "YearRemodAdd"] = 2020
    pd.testing.assert_frame_equal(house, original)


def test_outlier_boundaries_and_copies():
    X = pd.DataFrame({"GrLivArea": [4000, 4001, 4001]}, index=[7, 8, 9])
    y = pd.Series([100000, 300000, 299999], index=X.index)
    features, prices = remove_outliers(X, y)
    pd.testing.assert_frame_equal(features, X.iloc[:2])
    pd.testing.assert_series_equal(prices, y.iloc[:2])
    features.at[7, "GrLivArea"] = 10
    prices.at[7] = 100
    assert X.at[7, "GrLivArea"] == 4000 and y.at[7] == 100000


def read_houses(filename):
    path = Path(__file__).resolve().parents[1] / "data" / filename
    return pd.read_csv(path, na_values=["NA"], keep_default_na=False)


def test_garage_years_in_kaggle_test():
    houses = read_houses("test.csv").set_index("Id")
    years = Cleaner().transform(houses)["GarageYrBlt"]
    assert years.at[2593] == 2007 and years.at[2550] == 2008


def test_outlier_ids_in_kaggle_train():
    houses = read_houses("train.csv")
    features, prices = remove_outliers(houses.drop(columns="SalePrice"), houses["SalePrice"])
    assert set(houses["Id"]) - set(features["Id"]) == {524, 1299}
    assert features.index.equals(prices.index)


def test_rmse_uses_natural_log():
    assert rmse_log([1, 2], [np.e, 2]) == pytest.approx(np.sqrt(0.5))
