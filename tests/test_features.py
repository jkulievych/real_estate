"""
tests/test_features.py - Testy jednostkowe dla features.py

Author: Yuliia Kuliievych
License: MIT
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import FeatureExtractor, KEYWORDS


# Testy FeatureExtractor.extract

class TestExtract:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_has_parking_detected(self):
        result = self.extractor.extract("dispune de parcare subterana")
        assert result["has_parking"] == 1

    def test_has_parking_not_detected(self):
        result = self.extractor.extract("apartament cu 2 camere")
        assert result["has_parking"] == 0

    def test_has_balcony_detected(self):
        result = self.extractor.extract("balcon de 10 mp orientat spre sud")
        assert result["has_balcony"] == 1

    def test_has_elevator_lift(self):
        result = self.extractor.extract("bloc cu lift ultrasilentios")
        assert result["has_elevator"] == 1

    def test_has_elevator_ascensor(self):
        result = self.extractor.extract("ascensor de ultima generatie")
        assert result["has_elevator"] == 1

    def test_has_pool_detected(self):
        result = self.extractor.extract("complex cu piscina olimpica")
        assert result["has_pool"] == 1

    def test_has_metro_detected(self):
        result = self.extractor.extract("aproape de statia de metrou")
        assert result["has_metro"] == 1

    def test_has_ac_detected(self):
        result = self.extractor.extract("aer conditionat Daikin inclus")
        assert result["has_ac"] == 1

    def test_has_terrace_detected(self):
        result = self.extractor.extract("terasa generoasa de 20 mp")
        assert result["has_terrace"] == 1

    def test_has_garden_detected(self):
        result = self.extractor.extract("gradina privata de 100 mp")
        assert result["has_garden"] == 1

    def test_has_new_building_detected(self):
        result = self.extractor.extract("bloc nou finalizat in 2024")
        assert result["has_new_building"] == 1

    def test_has_own_heating_detected(self):
        result = self.extractor.extract("centrala termica proprie Vaillant")
        assert result["has_own_heating"] == 1

    def test_empty_description(self):
        result = self.extractor.extract("")
        assert all(v == 0 for v in result.values())

    def test_all_keys_present(self):
        result = self.extractor.extract("test")
        assert set(result.keys()) == set(KEYWORDS.keys())

    def test_case_insensitive(self):
        result = self.extractor.extract("PARCARE SUBTERANA")
        assert result["has_parking"] == 1

    def test_returns_binary(self):
        result = self.extractor.extract("parcare balcon lift piscina")
        assert all(v in (0, 1) for v in result.values())


# Testy FeatureExtractor.extract_desc_length

class TestExtractDescLength:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_basic(self):
        assert self.extractor.extract_desc_length("hello") == 5

    def test_empty(self):
        assert self.extractor.extract_desc_length("") == 0

    def test_strips_whitespace(self):
        assert self.extractor.extract_desc_length("  abc  ") == 3

    def test_longer_text(self):
        text = "a" * 500
        assert self.extractor.extract_desc_length(text) == 500


# Testy z niestandardowymi słownikami

class TestCustomKeywords:
    def test_custom_keyword(self):
        extractor = FeatureExtractor(keywords={"has_garden": r"\bogrod\b|garden"})
        assert extractor.extract("duzy ogrod prywatny")["has_garden"] == 1
        assert extractor.extract("brak zieleni")["has_garden"] == 0

    def test_empty_keywords(self):
        extractor = FeatureExtractor(keywords={})
        assert extractor.extract("dowolny tekst") == {}

    def test_multiple_matches_still_binary(ids=["park", "lift", "pool"]):
        extractor = FeatureExtractor()
        result = extractor.extract("parcare parcare parcare")
        assert result["has_parking"] == 1