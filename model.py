"""
model.py - Budowa i zapis modeli regresji ceny nieruchomości.

Modele:
    - Regresja liniowa (LinearRegression)
    - Las losowy (RandomForestRegressor)

Dane wejściowe:  data/processed/imobiliare.csv
Dane wyjściowe:  models/linear_model.pkl
                 models/rf_model.pkl
                 models/metadata.json

Author: ...
License: MIT
"""

import csv
import json
import math
import os
import pickle
from collections import defaultdict


# ── Stałe ─────────────────────────────────────────────────────────────────────

INPUT_FILE = "data/processed/imobiliare.csv"
MODELS_DIR = "models"

FEATURES = [
    "area_m2", "rooms",
    "layout_garsoniera", "layout_nedecomandat",
    "layout_nedefinit", "layout_penthouse", "layout_semidecomandat",
    "has_parking", "has_balcony", "has_elevator", "has_pool",
    "has_metro", "has_ac", "has_terrace", "has_garden",
    "has_new_building", "has_own_heating", "desc_length",
]
TARGET = "price_eur"

# Filtry usuwające outliery
PRICE_MIN = 10_000
PRICE_MAX = 2_000_000
AREA_MIN  = 10
AREA_MAX  = 500


# ── Pomocnicze funkcje matematyczne ──────────────────────────────────────────

def mean(values: list) -> float:
    """Zwraca średnią arytmetyczną listy."""
    return sum(values) / len(values)


def variance(values: list) -> float:
    """Zwraca wariancję listy."""
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def std(values: list) -> float:
    """Zwraca odchylenie standardowe listy."""
    return math.sqrt(variance(values))


def pearson(x: list, y: list) -> float:
    """Oblicza współczynnik korelacji Pearsona między x i y."""
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) *
                    sum((yi - my) ** 2 for yi in y))
    return num / den if den else 0.0


def r2_score(y_true: list, y_pred: list) -> float:
    """Oblicza współczynnik determinacji R²."""
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean(y_true)) ** 2 for t in y_true)
    return 1 - ss_res / ss_tot if ss_tot else 0.0


def rmse(y_true: list, y_pred: list) -> float:
    """Oblicza pierwiastek błędu średniokwadratowego (RMSE)."""
    return math.sqrt(mean([(t - p) ** 2 for t, p in zip(y_true, y_pred)]))


def mae(y_true: list, y_pred: list) -> float:
    """Oblicza średni błąd bezwzględny (MAE)."""
    return mean([abs(t - p) for t, p in zip(y_true, y_pred)])


# ── Ładowanie i czyszczenie danych ────────────────────────────────────────────

class DataLoader:
    """Klasa do ładowania i czyszczenia danych z pliku CSV."""

    def __init__(self, filepath: str):
        """
        Args:
            filepath: Ścieżka do pliku CSV.
        """
        self.filepath = filepath

    def load(self) -> tuple[list, list]:
        """
        Ładuje dane, usuwa outliery i zwraca X, y.

        Returns:
            Tuple (X, y) gdzie X to lista wektorów cech, y to lista cen.
        """
        rows = list(csv.DictReader(open(self.filepath, encoding="utf-8-sig")))

        X, y = [], []
        skipped = 0
        for r in rows:
            try:
                price = float(r[TARGET])
                area  = float(r["area_m2"])
                if not (PRICE_MIN <= price <= PRICE_MAX):
                    skipped += 1
                    continue
                if not (AREA_MIN <= area <= AREA_MAX):
                    skipped += 1
                    continue
                x_row = [float(r[f]) for f in FEATURES]
                X.append(x_row)
                y.append(price)
            except (ValueError, KeyError):
                skipped += 1

        print(f"Załadowano {len(X)} rekordów, pominięto {skipped}")
        return X, y

    def train_test_split(self, X: list, y: list,
                         test_size: float = 0.2,
                         seed: int = 42) -> tuple:
        """
        Dzieli dane na zbiór treningowy i testowy.

        Args:
            X: Lista wektorów cech.
            y: Lista wartości docelowych.
            test_size: Udział zbioru testowego (0-1).
            seed: Ziarno losowości.

        Returns:
            Tuple (X_train, X_test, y_train, y_test).
        """
        import random
        random.seed(seed)
        indices = list(range(len(X)))
        random.shuffle(indices)
        split = int(len(X) * (1 - test_size))
        train_idx = indices[:split]
        test_idx  = indices[split:]
        X_train = [X[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test  = [y[i] for i in test_idx]
        return X_train, X_test, y_train, y_test


# ── Regresja liniowa ──────────────────────────────────────────────────────────

class LinearRegressionModel:
    """
    Regresja liniowa metodą najmniejszych kwadratów (OLS).
    Implementacja bez zewnętrznych bibliotek ML.
    """

    def __init__(self):
        """Inicjalizacja modelu."""
        self.coefficients = []
        self.intercept    = 0.0
        self.feature_names = FEATURES

    def _transpose(self, matrix: list) -> list:
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    def _matmul(self, A: list, B: list) -> list:
        result = [[0.0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def _inverse(self, matrix: list) -> list:
        """Odwrócenie macierzy metodą Gaussa-Jordana."""
        n = len(matrix)
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
               for i, row in enumerate(matrix)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            div = aug[col][col]
            if abs(div) < 1e-12:
                raise ValueError("Macierz osobliwa — nie można odwrócić.")
            aug[col] = [x / div for x in aug[col]]
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][k] - factor * aug[col][k]
                                for k in range(2 * n)]
        return [row[n:] for row in aug]

    def fit(self, X: list, y: list) -> None:
        """
        Dopasowuje model do danych treningowych (OLS).

        Args:
            X: Lista wektorów cech.
            y: Lista wartości docelowych.
        """
        # Dodaj kolumnę jedynek (intercept)
        X_b = [[1.0] + row for row in X]
        Xt   = self._transpose(X_b)
        XtX  = self._matmul(Xt, X_b)
        Xty  = [[sum(Xt[i][k] * y[k] for k in range(len(y)))]
                for i in range(len(Xt))]
        inv  = self._inverse(XtX)
        beta = self._matmul(inv, Xty)
        self.intercept    = beta[0][0]
        self.coefficients = [beta[i + 1][0] for i in range(len(FEATURES))]

    def predict(self, X: list) -> list:
        """
        Przewiduje ceny dla podanych wektorów cech.

        Args:
            X: Lista wektorów cech.

        Returns:
            Lista przewidywanych cen.
        """
        preds = []
        for row in X:
            val = self.intercept + sum(c * x for c, x in
                                       zip(self.coefficients, row))
            preds.append(max(val, 0.0))
        return preds

    def predict_one(self, x: list) -> float:
        """
        Przewiduje cenę dla jednego wektora cech.

        Args:
            x: Wektor cech.

        Returns:
            Przewidywana cena.
        """
        return self.predict([x])[0]


# ── Las losowy (uproszczony) ──────────────────────────────────────────────────

class RandomForestModel:
    """
    Uproszczony las losowy oparty na sklearn.
    Używa sklearn tylko do treningu, reszta własna.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators: Liczba drzew w lesie.
            random_state: Ziarno losowości.
        """
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self._model        = None
        self.feature_names = FEATURES

    def fit(self, X: list, y: list) -> None:
        """
        Trenuje model lasu losowego.

        Args:
            X: Lista wektorów cech.
            y: Lista wartości docelowych.
        """
        from sklearn.ensemble import RandomForestRegressor
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: list) -> list:
        """
        Przewiduje ceny dla podanych wektorów cech.

        Args:
            X: Lista wektorów cech.

        Returns:
            Lista przewidywanych cen.
        """
        return [max(float(p), 0.0) for p in self._model.predict(X)]

    def predict_one(self, x: list) -> float:
        """
        Przewiduje cenę dla jednego wektora cech.

        Args:
            x: Wektor cech.

        Returns:
            Przewidywana cena.
        """
        return self.predict([x])[0]


# ── Trening i zapis ───────────────────────────────────────────────────────────

class ModelTrainer:
    """Klasa zarządzająca treningiem, ewaluacją i zapisem modeli."""

    def __init__(self, input_file: str = INPUT_FILE,
                 models_dir: str = MODELS_DIR):
        """
        Args:
            input_file: Ścieżka do przetworzonego CSV.
            models_dir: Katalog do zapisu modeli.
        """
        self.input_file = input_file
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def evaluate(self, model, X_test: list, y_test: list,
                 name: str) -> dict:
        """
        Ewaluuje model i wypisuje metryki.

        Args:
            model: Wytrenowany model z metodą predict().
            X_test: Testowe wektory cech.
            y_test: Testowe wartości docelowe.
            name: Nazwa modelu do wydruku.

        Returns:
            Słownik z metrykami: r2, rmse, mae.
        """
        preds = model.predict(X_test)
        metrics = {
            "r2":   round(r2_score(y_test, preds), 4),
            "rmse": round(rmse(y_test, preds), 2),
            "mae":  round(mae(y_test, preds), 2),
        }
        print(f"\n── {name} ──")
        print(f"  R²:   {metrics['r2']}")
        print(f"  RMSE: {metrics['rmse']:,.0f} EUR")
        print(f"  MAE:  {metrics['mae']:,.0f} EUR")
        return metrics

    def save(self, model, filename: str) -> None:
        """
        Zapisuje model do pliku pickle.

        Args:
            model: Wytrenowany model.
            filename: Nazwa pliku (bez ścieżki).
        """
        path = os.path.join(self.models_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Zapisano → {path}")

    def run(self) -> None:
        """Uruchamia pełny pipeline: ładowanie, trening, ewaluacja, zapis."""
        # Dane
        loader = DataLoader(self.input_file)
        X, y   = loader.load()
        X_train, X_test, y_train, y_test = loader.train_test_split(X, y)
        print(f"Trening: {len(X_train)} | Test: {len(X_test)}")

        # Statystyki opisowe
        print(f"\nCena — średnia: {mean(y):.0f} EUR | "
              f"std: {std(y):.0f} | "
              f"min: {min(y):.0f} | max: {max(y):.0f}")

        metrics = {}

        # Regresja liniowa
        lr = LinearRegressionModel()
        lr.fit(X_train, y_train)
        m_lr = self.evaluate(lr, X_test, y_test, "Regresja liniowa")
        metrics["linear"] = m_lr
        self.save(lr, "linear_model.pkl")

        # Współczynniki
        print("\n  Współczynniki regresji liniowej:")
        pairs = sorted(zip(FEATURES, lr.coefficients),
                       key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in pairs[:10]:
            print(f"    {feat:30}: {coef:+.2f}")

        # Las losowy
        try:
            rf = RandomForestModel()
            rf.fit(X_train, y_train)
            m_rf = self.evaluate(rf, X_test, y_test, "Las losowy")
            metrics["random_forest"] = m_rf
            self.save(rf, "rf_model.pkl")
        except ImportError:
            print("\nsklearn niedostępny — pominięto las losowy.")

        # Metadane
        meta = {
            "features": FEATURES,
            "target": TARGET,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "price_min": PRICE_MIN,
            "price_max": PRICE_MAX,
            "area_min": AREA_MIN,
            "area_max": AREA_MAX,
            "metrics": metrics,
        }
        meta_path = os.path.join(self.models_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"\nMetadane → {meta_path}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()