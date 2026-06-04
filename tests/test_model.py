"""
tests/test_model.py - Testy jednostkowe dla model.py

Author: Yuliia Kuliievych
License: MIT
"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import (
    mean, variance, std, pearson,
    r2_score, rmse, mae,
    LinearRegressionModel, DataLoader,
)

# Dane testowe — 30 wierszy, 18 cech (wymagane przez LinearRegressionModel)
TEST_X = [
    [323.3, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [230.1, 3, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [367.6, 5, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    [321.5, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    [412.7, 3, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    [187.4, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    [64.7,  4, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [465.3, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    [205.5, 3, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [431.2, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [238.8, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [227.1, 5, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    [139.9, 4, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [427.2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [339.1, 3, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    [224.7, 5, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    [133.0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [419.0, 5, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [488.3, 2, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    [159.5, 2, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [28.7,  5, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    [315.3, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [113.0, 5, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    [22.0,  2, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [34.6,  2, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [144.6, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    [275.9, 4, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [223.6, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [195.8, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [475.3, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
]
TEST_Y = [
    235777.0, 165159.0, 231605.0, 125061.0, 238294.0,
    89666.0,  168420.0, 216114.0, 98570.0,  189439.0,
    110317.0, 186852.0, 211106.0, 209245.0, 163187.0,
    58214.0,  156549.0, 182304.0, 63791.0,  179509.0,
    52530.0,  134012.0, 148040.0, 136705.0, 122438.0,
    228315.0, 141948.0, 80598.0,  85227.0,  154349.0,
]


# Testy funkcji matematycznych

class TestMean:
    def test_basic(self):
        assert mean([1, 2, 3, 4, 5]) == 3.0

    def test_single(self):
        assert mean([7]) == 7.0

    def test_floats(self):
        assert abs(mean([1.5, 2.5]) - 2.0) < 1e-9


class TestVariance:
    def test_zero(self):
        assert variance([5, 5, 5]) == 0.0

    def test_known(self):
        assert abs(variance([2, 4, 4, 4, 5, 5, 7, 9]) - 4.0) < 1e-9


class TestStd:
    def test_zero(self):
        assert std([3, 3, 3]) == 0.0

    def test_positive(self):
        assert std([1, 2, 3, 4, 5]) > 0


class TestPearson:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(pearson(x, y) - 1.0) < 1e-9

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert abs(pearson(x, y) + 1.0) < 1e-9

    def test_range(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 1, 4, 2, 3]
        assert -1.0 <= pearson(x, y) <= 1.0


class TestR2Score:
    def test_perfect(self):
        y = [1.0, 2.0, 3.0]
        assert abs(r2_score(y, y) - 1.0) < 1e-9

    def test_baseline(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [mean(y_true)] * 3
        assert abs(r2_score(y_true, y_pred)) < 1e-9

    def test_negative(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [3.0, 2.0, 1.0]
        assert r2_score(y_true, y_pred) < 0


class TestRmse:
    def test_zero(self):
        y = [1.0, 2.0, 3.0]
        assert rmse(y, y) == 0.0

    def test_known(self):
        y_true = [0.0, 0.0]
        y_pred = [1.0, 1.0]
        assert abs(rmse(y_true, y_pred) - 1.0) < 1e-9


class TestMae:
    def test_zero(self):
        y = [1.0, 2.0, 3.0]
        assert mae(y, y) == 0.0

    def test_known(self):
        y_true = [0.0, 0.0, 0.0]
        y_pred = [1.0, 2.0, 3.0]
        assert abs(mae(y_true, y_pred) - 2.0) < 1e-9


# Testy LinearRegressionModel

class TestLinearRegressionModel:
    def test_fit_returns_predictions(self):
        """Model powinien zwrócić predykcje dla danych treningowych."""
        model = LinearRegressionModel()
        model.fit(TEST_X, TEST_Y)
        preds = model.predict(TEST_X)
        assert len(preds) == len(TEST_Y)
        assert all(isinstance(p, float) for p in preds)

    def test_predict_one_returns_float(self):
        model = LinearRegressionModel()
        model.fit(TEST_X, TEST_Y)
        result = model.predict_one(TEST_X[0])
        assert isinstance(result, float)

    def test_predict_non_negative(self):
        """Przewidywania nie mogą być ujemne."""
        model = LinearRegressionModel()
        model.fit(TEST_X, TEST_Y)
        neg_row = [-999.0] + [0.0] * 17
        preds = model.predict([neg_row])
        assert preds[0] >= 0.0

    def test_coefficients_length(self):
        model = LinearRegressionModel()
        model.fit(TEST_X, TEST_Y)
        assert len(model.coefficients) == 18

    def test_intercept_is_float(self):
        model = LinearRegressionModel()
        model.fit(TEST_X, TEST_Y)
        assert isinstance(model.intercept, float)

    def test_r2_positive(self):
        """R² na danych treningowych powinno być dodatnie."""
        model = LinearRegressionModel()
        model.fit(TEST_X, TEST_Y)
        preds = model.predict(TEST_X)
        assert r2_score(TEST_Y, preds) > 0


# Testy DataLoader

class TestDataLoaderSplit:
    def test_split_sizes(self):
        X = [[i] for i in range(100)]
        y = list(range(100))
        loader = DataLoader("dummy.csv")
        X_train, X_test, y_train, y_test = loader.train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_split_no_overlap(self):
        X = [[i] for i in range(50)]
        y = list(range(50))
        loader = DataLoader("dummy.csv")
        X_train, X_test, y_train, y_test = loader.train_test_split(X, y)
        train_vals = set(x[0] for x in X_train)
        test_vals  = set(x[0] for x in X_test)
        assert train_vals.isdisjoint(test_vals)

    def test_split_reproducible(self):
        X = [[i] for i in range(100)]
        y = list(range(100))
        loader = DataLoader("dummy.csv")
        r1 = loader.train_test_split(X, y, seed=42)
        r2 = loader.train_test_split(X, y, seed=42)
        assert r1[0] == r2[0]

    def test_split_covers_all_data(self):
        X = [[i] for i in range(100)]
        y = list(range(100))
        loader = DataLoader("dummy.csv")
        X_train, X_test, y_train, y_test = loader.train_test_split(X, y)
        assert len(X_train) + len(X_test) == 100