"""
app.py - Aplikacja GUI do szacowania cen nieruchomości (Flet).

Author: ...
License: MIT
"""

import json
import os
import pickle
import sqlite3
from datetime import datetime

import flet as ft
from model import LinearRegressionModel, RandomForestModel  # noqa: F401

# ── Stałe ─────────────────────────────────────────────────────────────────────

MODELS_DIR    = "models"
DB_PATH       = "data/history.db"
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

LAYOUT_OPTIONS = [
    "decomandat", "semidecomandat", "nedecomandat",
    "garsoniera", "penthouse", "nedefinit",
]

# Paleta strawberry & matcha
BG         = "#838F58"   # matcha — tło
SURFACE    = "#6B7848"   # karty
SURFACE2   = "#7A8750"   # inputy / tagi
GREEN      = "#F9D1D9"   # strawberry — akcent
GREEN_DARK = "#F2B8C2"   # ciemniejszy róż
TEXT       = "#F9EAD2"   # jasny tekst
TEXT_MUTED = "#C8D4A0"   # przygaszony

PRICE_RANGES = [
    (0,       80_000,  "Okazja",     "#B8E6A0"),
    (80_000,  150_000, "Przecietna", "#F9EAD2"),
    (150_000, 999_999, "Wysoka",     "#F9D1D9"),
]


# ── Baza danych ───────────────────────────────────────────────────────────────

class Database:
    """Klasa do zarządzania historią oszacowań w SQLite."""

    def __init__(self, path: str = DB_PATH):
        """
        Args:
            path: Ścieżka do pliku bazy danych.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._init()

    def _init(self) -> None:
        """Tworzy tabelę historii jeśli nie istnieje."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT,
                    area_m2      REAL,
                    rooms        INTEGER,
                    layout       TEXT,
                    has_parking  INTEGER,
                    has_balcony  INTEGER,
                    has_elevator INTEGER,
                    has_metro    INTEGER,
                    model_used   TEXT,
                    price_lr     REAL,
                    price_rf     REAL
                )
            """)

    def save(self, params: dict, price_lr: float, price_rf: float) -> None:
        """
        Zapisuje wynik oszacowania do bazy.

        Args:
            params: Słownik parametrów.
            price_lr: Wynik regresji liniowej.
            price_rf: Wynik lasu losowego.
        """
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                INSERT INTO history
                (timestamp, area_m2, rooms, layout, has_parking,
                 has_balcony, has_elevator, has_metro, model_used,
                 price_lr, price_rf)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                params["area_m2"], params["rooms"], params["layout"],
                params["has_parking"], params["has_balcony"],
                params["has_elevator"], params["has_metro"],
                "both", price_lr, price_rf,
            ))

    def fetch_all(self) -> list:
        """
        Pobiera całą historię oszacowań.

        Returns:
            Lista krotek z historią.
        """
        with sqlite3.connect(self.path) as conn:
            return conn.execute(
                "SELECT * FROM history ORDER BY id DESC"
            ).fetchall()

    def clear(self) -> None:
        """Usuwa całą historię oszacowań."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM history")


# ── Ładowanie modeli ──────────────────────────────────────────────────────────

class ModelLoader:
    """Klasa do ładowania wytrenowanych modeli."""

    def __init__(self, models_dir: str = MODELS_DIR):
        """
        Args:
            models_dir: Katalog z plikami modeli.
        """
        self.models_dir = models_dir
        self.linear = None
        self.rf     = None
        self.meta   = {}
        self._load()

    def _load(self) -> None:
        """Ładuje modele i metadane z plików pickle/json."""
        with open(METADATA_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(self.models_dir, "linear_model.pkl"), "rb") as f:
            self.linear = pickle.load(f)
        with open(os.path.join(self.models_dir, "rf_model.pkl"), "rb") as f:
            self.rf = pickle.load(f)

    def predict(self, feature_vector: list) -> tuple:
        """
        Zwraca przewidywania obu modeli.

        Args:
            feature_vector: Lista wartości cech.

        Returns:
            Tuple (cena_lr, cena_rf).
        """
        return (
            self.linear.predict_one(feature_vector),
            self.rf.predict_one(feature_vector),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_feature_vector(params: dict, features: list) -> list:
    """
    Buduje wektor cech z parametrów użytkownika.

    Args:
        params: Słownik parametrów.
        features: Lista nazw cech z metadata.json.

    Returns:
        Lista wartości cech w odpowiedniej kolejności.
    """
    layout = params.get("layout", "decomandat")
    row = {
        "area_m2":               params.get("area_m2", 50),
        "rooms":                 params.get("rooms", 2),
        "layout_garsoniera":     1 if layout == "garsoniera" else 0,
        "layout_nedecomandat":   1 if layout == "nedecomandat" else 0,
        "layout_nedefinit":      1 if layout == "nedefinit" else 0,
        "layout_penthouse":      1 if layout == "penthouse" else 0,
        "layout_semidecomandat": 1 if layout == "semidecomandat" else 0,
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
    """
    Zwraca etykietę i kolor dla danej ceny.

    Args:
        price: Cena w EUR.

    Returns:
        Tuple (etykieta, kolor).
    """
    for lo, hi, label, color in PRICE_RANGES:
        if lo <= price < hi:
            return label, color
    return "Wysoka", "#C0525A"


# ── Główna aplikacja ──────────────────────────────────────────────────────────

class RealEstateApp:
    """Główna klasa aplikacji Flet do szacowania cen nieruchomości."""

    def __init__(self, page: ft.Page):
        """
        Args:
            page: Obiekt strony Flet.
        """
        self.page     = page
        self.db       = Database()
        self.models   = ModelLoader()
        self.features = self.models.meta["features"]
        self._setup_page()
        self._build_ui()

    def _setup_page(self) -> None:
        """Konfiguruje stronę Flet."""
        self.page.title            = "Estymator Cen Nieruchomości"
        self.page.theme_mode       = ft.ThemeMode.LIGHT
        self.page.window.width     = 960
        self.page.window.height    = 780
        self.page.window.min_width = 640
        self.page.bgcolor          = BG
        self.page.padding          = 0

    def _field(self, label: str, value: str, width: int) -> ft.TextField:
        """
        Tworzy stylizowane pole tekstowe.

        Args:
            label: Etykieta pola.
            value: Domyślna wartość.
            width: Szerokość pola.

        Returns:
            Obiekt TextField.
        """
        return ft.TextField(
            label=label,
            label_style=ft.TextStyle(color=TEXT_MUTED, size=12),
            value=value,
            keyboard_type=ft.KeyboardType.NUMBER,
            width=width,
            border_radius=8,
            bgcolor=SURFACE2,
            color=TEXT,
            border_color="#D4B8BC",
            focused_border_color=GREEN,
            cursor_color=GREEN,
            on_change=self._validate_number,
        )

    def _section(self, title: str) -> ft.Text:
        """
        Tworzy nagłówek sekcji.

        Args:
            title: Tytuł sekcji.

        Returns:
            Obiekt Text.
        """
        return ft.Text(
            title.upper(),
            size=11,
            weight=ft.FontWeight.W_700,
            color=GREEN,
        )

    def _card(self, content: ft.Control) -> ft.Container:
        """
        Opakowuje zawartość w stylizowaną kartę.

        Args:
            content: Zawartość karty.

        Returns:
            Obiekt Container.
        """
        return ft.Container(
            content=content,
            bgcolor=SURFACE,
            border_radius=16,
            padding=24,
            margin=16,
            shadow=ft.BoxShadow(
                blur_radius=16,
                color="#00000012",
                offset=ft.Offset(0, 4),
            ),
        )

    def _build_ui(self) -> None:
        """Buduje interfejs użytkownika."""
        self.area_field  = self._field("Powierzchnia (m2)", "50", 170)
        self.rooms_field = self._field("Liczba pokoi", "2", 140)

        self.layout_dd = ft.Dropdown(
            label="Typ układu",
            label_style=ft.TextStyle(color=TEXT_MUTED, size=12),
            width=200,
            border_radius=8,
            bgcolor=SURFACE2,
            color=TEXT,
            border_color="#D4B8BC",
            focused_border_color=GREEN,
            options=[ft.dropdown.Option(o) for o in LAYOUT_OPTIONS],
            value="decomandat",
        )

        amenity_labels = {
            "has_parking":      "Parking",
            "has_balcony":      "Balkon",
            "has_elevator":     "Winda",
            "has_metro":        "Metro",
            "has_ac":           "Klimatyzacja",
            "has_terrace":      "Taras",
            "has_pool":         "Basen",
            "has_garden":       "Ogrod",
            "has_new_building": "Nowy budynek",
            "has_own_heating":  "Własne ogrzewanie",
        }

        self.checks = {}

        def make_tag(key: str, label: str) -> ft.Container:
            cb = ft.Checkbox(
                value=False,
                fill_color=GREEN,
                check_color=SURFACE,
                active_color=GREEN,
            )
            cb.data = key
            self.checks[key] = cb
            return ft.Container(
                content=ft.Row([
                    cb,
                    ft.Text(label, size=13, color=TEXT),
                ], spacing=4),
                bgcolor=BG,
                border_radius=8,
                padding=8,
            )

        tags = ft.Row(
            [make_tag(k, v) for k, v in amenity_labels.items()],
            wrap=True,
            spacing=8,
            run_spacing=8,
        )

        self.result_lr    = ft.Text("—", size=30, weight=ft.FontWeight.BOLD, color=GREEN_DARK)
        self.result_rf    = ft.Text("—", size=30, weight=ft.FontWeight.BOLD, color=GREEN_DARK)
        self.result_label = ft.Text("", size=14, weight=ft.FontWeight.W_600, color=TEXT_MUTED)
        self.error_text   = ft.Text("", color=GREEN_DARK, size=12)

        estimate_btn = ft.ElevatedButton(
            "Szacuj cenę",
            on_click=self._estimate,
            style=ft.ButtonStyle(
                bgcolor=GREEN,
                color=SURFACE,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            icon=ft.Icons.ARROW_FORWARD,
        )

        self.history_tab = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Data",     color=TEXT_MUTED, size=12)),
                ft.DataColumn(ft.Text("m2",       color=TEXT_MUTED, size=12)),
                ft.DataColumn(ft.Text("Pokoje",   color=TEXT_MUTED, size=12)),
                ft.DataColumn(ft.Text("Układ",    color=TEXT_MUTED, size=12)),
                ft.DataColumn(ft.Text("LR (EUR)", color=TEXT_MUTED, size=12)),
                ft.DataColumn(ft.Text("RF (EUR)", color=TEXT_MUTED, size=12)),
            ],
            rows=[],
            heading_row_color=SURFACE2,
            border_radius=8,
            horizontal_lines=ft.BorderSide(width=1, color=SURFACE2),
        )

        clear_btn = ft.TextButton(
            "Wyczysc historię",
            on_click=self._clear_history,
            style=ft.ButtonStyle(color=TEXT_MUTED),
        )

        header = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text("Estymator Nieruchomości",
                            size=32, weight=ft.FontWeight.BOLD, color=GREEN_DARK),
                    ft.Text("Rumunia · imobiliare.ro",
                            size=14, color=TEXT_MUTED),
                ], spacing=2),
                ft.Text("RO", size=28, weight=ft.FontWeight.BOLD, color=GREEN),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            bgcolor=BG,
            padding=24,
        )

        form_card = self._card(ft.Column([
            self._section("Parametry"),
            ft.Row([self.area_field, self.rooms_field, self.layout_dd],
                   wrap=True, spacing=12),
            ft.Container(height=4),
            self._section("Udogodnienia"),
            tags,
            self.error_text,
            ft.Row([estimate_btn], alignment=ft.MainAxisAlignment.END),
        ], spacing=14))

        result_card = self._card(ft.Column([
            self._section("Wynik szacowania"),
            ft.Row([
                ft.Column([
                    ft.Text("Regresja liniowa", size=11, color=TEXT_MUTED),
                    self.result_lr,
                ], spacing=4),
                ft.Container(width=1, height=50, bgcolor="#6B7848"),
                ft.Column([
                    ft.Text("Las losowy", size=11, color=TEXT_MUTED),
                    self.result_rf,
                ], spacing=4),
                ft.Container(width=1, height=50, bgcolor="#6B7848"),
                ft.Column([
                    ft.Text("Ocena", size=11, color=TEXT_MUTED),
                    self.result_label,
                ], spacing=4),
            ], spacing=24),
        ], spacing=14))

        history_card = self._card(ft.Column([
            ft.Row([
                self._section("Historia oszacowań"),
                clear_btn,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            self.history_tab,
        ], spacing=12))

        self.page.add(
            header,
            ft.Column([
                form_card,
                result_card,
                history_card,
            ], scroll=ft.ScrollMode.AUTO, spacing=0, expand=True),
        )
        self._refresh_history()

    def _validate_number(self, e: ft.ControlEvent) -> None:
        """Usuwa niedozwolone znaki z pól numerycznych."""
        field = e.control
        cleaned = "".join(c for c in field.value if c.isdigit() or c == ".")
        if cleaned != field.value:
            field.value = cleaned
            field.update()

    def _get_params(self) -> dict | None:
        """
        Pobiera i waliduje parametry z formularza.

        Returns:
            Słownik parametrów lub None jeśli błąd walidacji.
        """
        self.error_text.value = ""
        try:
            area  = float(self.area_field.value or 0)
            rooms = int(self.rooms_field.value or 0)
        except ValueError:
            self.error_text.value = "Wprowadz poprawne wartosci liczbowe."
            self.error_text.update()
            return None

        meta = self.models.meta
        if not (meta["area_min"] <= area <= meta["area_max"]):
            self.error_text.value = (
                f"Powierzchnia musi byc miedzy "
                f"{meta['area_min']} a {meta['area_max']} m2."
            )
            self.error_text.update()
            return None
        if not (1 <= rooms <= 10):
            self.error_text.value = "Liczba pokoi musi byc miedzy 1 a 10."
            self.error_text.update()
            return None

        params = {
            "area_m2":     area,
            "rooms":       rooms,
            "layout":      self.layout_dd.value,
            "desc_length": 300,
        }
        for key, cb in self.checks.items():
            params[key] = 1 if cb.value else 0

        return params

    def _estimate(self, e: ft.ControlEvent) -> None:
        """Szacuje cenę i wyświetla wynik."""
        params = self._get_params()
        if params is None:
            return

        vec = build_feature_vector(params, self.features)
        price_lr, price_rf = self.models.predict(vec)

        avg = (price_lr + price_rf) / 2
        label, color = price_label(avg)

        self.result_lr.value    = f"EUR {price_lr:,.0f}"
        self.result_rf.value    = f"EUR {price_rf:,.0f}"
        self.result_label.value = label
        self.result_label.color = color

        self.result_lr.update()
        self.result_rf.update()
        self.result_label.update()

        self.db.save(params, price_lr, price_rf)
        self._refresh_history()

    def _refresh_history(self) -> None:
        """Odświeża tabelę historii."""
        rows = self.db.fetch_all()
        self.history_tab.rows = [
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(r[1])[:16], size=11, color=TEXT_MUTED)),
                ft.DataCell(ft.Text(str(r[2]), color=TEXT)),
                ft.DataCell(ft.Text(str(r[3]), color=TEXT)),
                ft.DataCell(ft.Text(str(r[4]), color=TEXT)),
                ft.DataCell(ft.Text(f"EUR {r[10]:,.0f}" if r[10] else "—", color=GREEN_DARK)),
                ft.DataCell(ft.Text(f"EUR {r[11]:,.0f}" if r[11] else "—", color=GREEN_DARK)),
            ])
            for r in rows[:20]
        ]
        self.history_tab.update()

    def _clear_history(self, e: ft.ControlEvent) -> None:
        """Czyści historię oszacowań."""
        self.db.clear()
        self._refresh_history()


# ── Punkt wejścia ─────────────────────────────────────────────────────────────

def main(page: ft.Page) -> None:
    """
    Punkt wejścia aplikacji Flet.

    Args:
        page: Obiekt strony Flet.
    """
    RealEstateApp(page)


if __name__ == "__main__":
    ft.app(target=main)