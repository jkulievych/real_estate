"""
tests/test_app.py - Testy jednostkowe dla funkcji pomocniczych z app.py

Author: Yuliia Kuliievych
License: MIT
"""

import sys
import os
import csv
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Kopiujemy testowane funkcje bezpośrednio ──────────────────────────────────
# (unikamy importu całego app.py z zależnością od flet/flet_lottie)

PRICE_RANGES = [
    (0,       80_000,  "Okazja",     "#B8E6A0"),
    (80_000,  150_000, "Przecietna", "#F9EAD2"),
    (150_000, 999_999, "Wysoka",     "#F9D1D9"),
]


def build_feature_vector(params: dict, features: list) -> list:
    layout = params.get("layout", "rozdzielny")
    row = {
        "area_m2":               params.get("area_m2", 50),
        "rooms":                 params.get("rooms", 2),
        "layout_garsoniera":     1 if layout == "kawalerka" else 0,
        "layout_nedecomandat":   1 if layout == "przechodny" else 0,
        "layout_nedefinit":      1 if layout == "nieokreślony" else 0,
        "layout_penthouse":      1 if layout == "penthouse" else 0,
        "layout_semidecomandat": 1 if layout == "półrozdzielny" else 0,
        "has_parking":           params.get("has_parking", 0),
        "has_balcony":           params.get("has_balcony", 0),
        "has_elevator":          params.get("has_elevator", 0),
        "has_pool":              params.get("has_pool", 0),
        "has_metro":             params.get("has_metro", 0),
        "has_ac":                params.get("has_ac", 0),
        "has_terrace":           params.get("has_terrace", 0),
        "has_garden":            params.get("has_garden", 0),
        "has_new_building":      params.get("has_new_building", 0),
        "has_own_heating":       params.get("has_own_heating", 0),
        "desc_length":           params.get("desc_length", 300),
    }
    return [row[f] for f in features]


def price_label(price: float) -> tuple:
    for lo, hi, label, color in PRICE_RANGES:
        if lo <= price < hi:
            return label, color
    return "Wysoka", "#F9D1D9"


class SimilarListingsFinder:
    def __init__(self, csv_path: str = "data/processed/imobiliare.csv"):
        self._listings = []
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                self._listings = list(csv.DictReader(f))
        except FileNotFoundError:
            pass

    def find(self, area: float, rooms: int, n: int = 3) -> list:
        def score(row):
            try:
                return abs(float(row["area_m2"]) - area) + abs(int(row["rooms"]) - rooms) * 10
            except (ValueError, KeyError):
                return 999999
        return sorted(self._listings, key=score)[:n]



#  Testy build_feature_vector

FEATURES = [
    "area_m2", "rooms",
    "layout_garsoniera", "layout_nedecomandat",
    "layout_nedefinit", "layout_penthouse", "layout_semidecomandat",
    "has_parking", "has_balcony", "has_elevator", "has_pool",
    "has_metro", "has_ac", "has_terrace", "has_garden",
    "has_new_building", "has_own_heating", "desc_length",
]


class TestBuildFeatureVector:
    def test_length(self):
        params = {"area_m2": 50, "rooms": 2, "layout": "rozdzielny"}
        vec = build_feature_vector(params, FEATURES)
        assert len(vec) == len(FEATURES)

    def test_area_rooms(self):
        params = {"area_m2": 75.5, "rooms": 3, "layout": "rozdzielny"}
        vec = build_feature_vector(params, FEATURES)
        assert vec[0] == 75.5
        assert vec[1] == 3

    def test_layout_garsoniera(self):
        params = {"area_m2": 30, "rooms": 1, "layout": "kawalerka"}
        vec = build_feature_vector(params, FEATURES)
        idx = FEATURES.index("layout_garsoniera")
        assert vec[idx] == 1

    def test_layout_semidecomandat(self):
        params = {"area_m2": 50, "rooms": 2, "layout": "półrozdzielny"}
        vec = build_feature_vector(params, FEATURES)
        idx = FEATURES.index("layout_semidecomandat")
        assert vec[idx] == 1

    def test_layout_decomandat_all_zeros(self):
        params = {"area_m2": 50, "rooms": 2, "layout": "rozdzielny"}
        vec = build_feature_vector(params, FEATURES)
        layout_features = [
            "layout_garsoniera", "layout_nedecomandat",
            "layout_nedefinit", "layout_penthouse", "layout_semidecomandat"
        ]
        for f in layout_features:
            assert vec[FEATURES.index(f)] == 0

    def test_binary_flags(self):
        params = {
            "area_m2": 50, "rooms": 2, "layout": "rozdzielny",
            "has_parking": 1, "has_balcony": 1, "has_elevator": 0,
        }
        vec = build_feature_vector(params, FEATURES)
        assert vec[FEATURES.index("has_parking")] == 1
        assert vec[FEATURES.index("has_balcony")] == 1
        assert vec[FEATURES.index("has_elevator")] == 0

    def test_defaults_for_missing_params(self):
        params = {"area_m2": 50, "rooms": 2, "layout": "rozdzielny"}
        vec = build_feature_vector(params, FEATURES)
        assert vec[FEATURES.index("has_parking")] == 0

    def test_only_one_layout_active(self, ids=["garsoniera", "niedecomandat",
                                               "nieokreslony", "penthouse", "semidecomandat"]):
        for layout in ["kawalerka", "przechodny", "nieokreślony", "penthouse", "półrozdzielny"]:
            params = {"area_m2": 50, "rooms": 2, "layout": layout}
            vec = build_feature_vector(params, FEATURES)
            layout_cols = [
                vec[FEATURES.index("layout_garsoniera")],
                vec[FEATURES.index("layout_nedecomandat")],
                vec[FEATURES.index("layout_nedefinit")],
                vec[FEATURES.index("layout_penthouse")],
                vec[FEATURES.index("layout_semidecomandat")],
            ]
            assert sum(layout_cols) <= 1


# Testy price_label

class TestPriceLabel:
    def test_okazja(self):
        label, _ = price_label(50_000)
        assert label == "Okazja"

    def test_przecietna(self):
        label, _ = price_label(100_000)
        assert label == "Przecietna"

    def test_wysoka(self):
        label, _ = price_label(200_000)
        assert label == "Wysoka"

    def test_returns_tuple(self):
        result = price_label(80_000)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_color_is_string(self):
        _, color = price_label(50_000)
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_boundary_80000(self):
        label_below, _ = price_label(79_999)
        label_above, _ = price_label(80_000)
        assert label_below == "Okazja"
        assert label_above == "Przecietna"

    def test_very_high_price(self):
        label, _ = price_label(5_000_000)
        assert label == "Wysoka"


# ── Testy SimilarListingsFinder ───────────────────────────────────────────────

class TestSimilarListingsFinder:
    def test_missing_file_no_crash(self):
        finder = SimilarListingsFinder(csv_path="nonexistent.csv")
        results = finder.find(50, 2)
        assert results == []

    def test_find_returns_list(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "id,title,area_m2,rooms,price_eur,url\n"
            "1,Apt A,50,2,100000,http://a\n"
            "2,Apt B,60,3,120000,http://b\n"
            "3,Apt C,45,2,90000,http://c\n",
            encoding="utf-8"
        )
        finder = SimilarListingsFinder(csv_path=str(csv_file))
        results = finder.find(50, 2, n=2)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_find_closest_area(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "id,title,area_m2,rooms,price_eur,url\n"
            "1,Apt A,50,2,100000,http://a\n"
            "2,Apt B,200,2,300000,http://b\n",
            encoding="utf-8"
        )
        finder = SimilarListingsFinder(csv_path=str(csv_file))
        results = finder.find(55, 2, n=1)
        assert results[0]["title"] == "Apt A"

    def test_find_n_results(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        rows = "\n".join(f"{i},Apt {i},{40+i},2,{90000+i*1000},http://{i}"
                         for i in range(10))
        csv_file.write_text(
            "id,title,area_m2,rooms,price_eur,url\n" + rows,
            encoding="utf-8"
        )
        finder = SimilarListingsFinder(csv_path=str(csv_file))
        results = finder.find(45, 2, n=3)
        assert len(results) == 3