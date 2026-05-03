"""
features.py - Ekstrakcja cech ze słownika opisów ogłoszeń nieruchomości.

Author: ...
License: MIT
"""

import csv
import re


# Słowniki słów kluczowych dla każdej cechy
KEYWORDS = {
    "has_parking": r"parcar",
    "has_balcony": r"balcon",
    "has_elevator": r"lift|ascensor",
    "has_pool": r"piscin",
    "has_metro": r"metrou|metro",
    "has_ac": r"aer condi[tț]ionat|climatiz",
    "has_terrace": r"tera[sș][aă]",
    "has_garden": r"gr[aă]din[aă]",
    "has_new_building": r"bloc nou|construc[tț]ie nou",
    "has_own_heating": r"central[aă]\s+proprie|central[aă]\s+termic",
}


class FeatureExtractor:
    """Klasa do ekstrakcji cech binarnych z opisów ogłoszeń."""

    def __init__(self, keywords: dict = KEYWORDS):
        """
        Inicjalizacja ekstraktora cech.

        Args:
            keywords: Słownik {nazwa_cechy: wzorzec_regex}.
        """
        self.keywords = keywords

    def extract(self, description: str) -> dict:
        """
        Wyciąga cechy binarne z opisu ogłoszenia.

        Args:
            description: Tekst opisu ogłoszenia.

        Returns:
            Słownik {nazwa_cechy: 0 lub 1}.
        """
        result = {}
        for feature, pattern in self.keywords.items():
            result[feature] = 1 if re.search(pattern, description, re.I) else 0
        return result

    def extract_desc_length(self, description: str) -> int:
        """
        Zwraca długość opisu w znakach.

        Args:
            description: Tekst opisu ogłoszenia.

        Returns:
            Liczba znaków w opisie.
        """
        return len(description.strip())

    def process_file(self, input_path: str, output_path: str) -> None:
        """
        Przetwarza plik CSV — dodaje kolumny z cechami i zapisuje wynik.

        Args:
            input_path: Ścieżka do pliku wejściowego CSV.
            output_path: Ścieżka do pliku wyjściowego CSV.
        """
        rows = list(csv.DictReader(open(input_path, encoding="utf-8-sig")))

        new_fields = list(self.keywords.keys()) + ["desc_length"]
        fields = list(rows[0].keys()) + new_fields

        for r in rows:
            desc = r.get("description", "")
            features = self.extract(desc)
            features["desc_length"] = self.extract_desc_length(desc)
            r.update(features)

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        counts = {f: sum(r[f] == "1" or r[f] == 1 for r in rows) for f in self.keywords}
        print(f"Przetworzono {len(rows)} rekordów → '{output_path}'")
        print("Częstość cech:")
        for feat, count in counts.items():
            print(f"  {feat:25}: {count:4} ({count / len(rows) * 100:.0f}%)")


if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process_file(
        input_path="data/raw/imobiliare.csv",
        output_path="data/processed/imobiliare.csv",
    )