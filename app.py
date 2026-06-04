"""
app.py - Aplikacja GUI do szacowania cen nieruchomości (Flet).

Author: Yuliia Kuliievych
License: MIT
"""

import json
import os
import pickle
import sqlite3
import threading
import time
from datetime import datetime
from flet_lottie import Lottie


import flet as ft
from model import LinearRegressionModel, RandomForestModel  # noqa: F401


MODELS_DIR    = "models"
DB_PATH       = "data/history.db"
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

LAYOUT_OPTIONS = [
    "rozdzielny", "półrozdzielny", "przechodny",
    "kawalerka", "penthouse", "nieokreślony",
]

LAYOUT_DISPLAY = {
    "rozdzielny":    "Rozdzielny",
    "półrozdzielny": "Półrozdzielny",
    "przechodny":    "Przechodny",
    "kawalerka":     "Kawalerka",
    "penthouse":     "Penthouse",
    "nieokreślony":  "Nieokreślony",
    "decomandat":    "Rozdzielny",
    None:            "—",
    "":              "—",
}

BG         = "#3D5D91"
SURFACE    = "#5A86CB"
SURFACE2   = "#4A6EA8"
GREEN      = "#F2AEBC"
GREEN_DARK = "#F2AEBC"
TEXT       = "#F2DCDB"
TEXT_MUTED = "#C9D8EE"

PRICE_RANGES = [
    (0,       80_000,  "Okazja",     "#B8E6A0"),
    (80_000,  150_000, "Przeciętna", "#F9EAD2"),
    (150_000, 999_999, "Wysoka",     "#F9D1D9"),
]



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



def build_feature_vector(params: dict, features: list) -> list:
    """
    Buduje wektor cech z parametrów użytkownika.

    Args:
        params: Słownik parametrów.
        features: Lista nazw cech z metadata.json.

    Returns:
        Lista wartości cech w odpowiedniej kolejności.
    """
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
    return "Wysoka", "#F9D1D9"


class SimilarListingsFinder:
    """Klasa do wyszukiwania podobnych ogłoszeń z CSV."""

    def __init__(self, csv_path: str = "data/processed/imobiliare.csv"):
        """
        Args:
            csv_path: Ścieżka do pliku CSV z ogłoszeniami.
        """
        import csv
        self._listings = []
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                self._listings = list(csv.DictReader(f))
        except FileNotFoundError:
            pass

    def find(self, area: float, rooms: int, n: int = 3) -> list:
        """
        Znajduje n najbardziej podobnych ogłoszeń.

        Args:
            area: Powierzchnia w m2.
            rooms: Liczba pokoi.
            n: Liczba wyników.

        Returns:
            Lista słowników z ogłoszeniami.
        """
        def score(row):
            try:
                return abs(float(row["area_m2"]) - area) + abs(int(row["rooms"]) - rooms) * 10
            except (ValueError, KeyError):
                return 999999

        ranked = sorted(self._listings, key=score)
        return ranked[:n]


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
        self._btn_animating = False
        self._cards_to_reveal = []
        self._setup_page()
        self._build_ui()
        self.finder = SimilarListingsFinder()

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
            border_color="#555533",
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

    def _animated_card(self, content: ft.Control, delay: float = 0) -> ft.Container:
        """
        Opakowuje kartę w stylizowany kontener.

        Args:
            content: Zawartość karty.
            delay: Nieużywane, zachowane dla kompatybilności.

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
                color="#00000018",
                offset=ft.Offset(0, 4),
            ),
        )

    def _animate_number(self, text_ctrl: ft.Text, target: float) -> None:
        """
        Animuje narastanie liczby od 0 do wartości docelowej.

        Args:
            text_ctrl: Kontrolka Text do zaktualizowania.
            target: Docelowa wartość liczbowa.
        """
        steps  = 1
        delay  = 0

        def run():
            for i in range(1, steps + 1):
                val = target * (i / steps)
                text_ctrl.value = f"EUR {val:,.0f}"
                text_ctrl.update()
                time.sleep(delay)

        threading.Thread(target=run, daemon=True).start()

    def _pulse_button(self, btn: ft.ElevatedButton) -> None:
        """
        Animuje pulsowanie przycisku po kliknięciu (scale up → down).

        Args:
            btn: Przycisk do animacji.
        """
        if self._btn_animating:
            return
        self._btn_animating = True

        def run():
            btn.scale = ft.Scale(1.08)
            btn.update()
            time.sleep(0.12)
            btn.scale = ft.Scale(1.0)
            btn.update()
            self._btn_animating = False
        self._cards_to_reveal = []

        threading.Thread(target=run, daemon=True).start()

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
            border_color="#555533",
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
            "has_garden":       "Ogród",
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
                content=ft.Row([cb, ft.Text(label, size=13, color=TEXT)], spacing=4),
                bgcolor=BG,
                border_radius=8,
                padding=8,
            )

        tags = ft.Row(
            [make_tag(k, v) for k, v in amenity_labels.items()],
            wrap=True, spacing=8, run_spacing=8,
        )

        self.result_lr    = ft.Text("—", size=30, weight=ft.FontWeight.BOLD, color=GREEN_DARK)
        self.result_rf    = ft.Text("—", size=30, weight=ft.FontWeight.BOLD, color=GREEN_DARK)
        self.result_label = ft.Text("", size=14, weight=ft.FontWeight.W_600, color=TEXT_MUTED)
        self.error_text   = ft.Text("", color=GREEN_DARK, size=12)
        self.similar_col = ft.Column([], spacing=8, visible=False)

        self._spinner = ft.ProgressRing(
            width=16, height=16, stroke_width=2,
            color=SURFACE, visible=False,
        )
        self._estimate_btn = ft.ElevatedButton(
            "Szacuj cenę",
            on_click=self._estimate,
            style=ft.ButtonStyle(
                bgcolor=GREEN,
                color=SURFACE,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            icon=ft.Icons.ARROW_FORWARD,
            animate_scale=150,
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
            "Wyczyść historię",
            on_click=self._clear_history,
            style=ft.ButtonStyle(color=TEXT_MUTED),
        )

        lottie = Lottie(
            src="animation.json",
            width=200,
            height=200,
            repeat=True,
            animate=True,
        )

        header = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text("Estymator Nieruchomości",
                            size=32, weight=ft.FontWeight.BOLD, color=GREEN_DARK),
                    ft.Text("Rumunia · imobiliare.ro",
                            size=14, color=TEXT_MUTED),
                ], spacing=2),
                lottie,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            bgcolor=BG,
            padding=24,
        )

        self._form_card = self._animated_card(ft.Column([
            self._section("Parametry"),
            ft.Row([self.area_field, self.rooms_field, self.layout_dd],
                   wrap=True, spacing=12),
            ft.Container(height=4),
            self._section("Udogodnienia"),
            tags,
            self.error_text,
            ft.Row([self._estimate_btn], alignment=ft.MainAxisAlignment.END),
        ], spacing=14), delay=0.1)

        self._result_card = self._animated_card(ft.Column([
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
        ], spacing=14), delay=0.25)

        self._history_card = self._animated_card(ft.Column([
            ft.Row([
                self._section("Historia oszacowań"),
                clear_btn,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            self.history_tab,
        ], spacing=12), delay=0.4)

        self._main_col = ft.Column([
            self._form_card,
            self._result_card,
            self._history_card,
        ], scroll=ft.ScrollMode.AUTO, spacing=0, expand=True)
        self.page.add(header, self._main_col)
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
            self.error_text.value = "Wprowadź poprawne wartości liczbowe."
            self.error_text.update()
            return None

        meta = self.models.meta
        if not (meta["area_min"] <= area <= meta["area_max"]):
            self.error_text.value = (
                f"Powierzchnia musi byc między "
                f"{meta['area_min']} a {meta['area_max']} m2."
            )
            self.error_text.update()
            return None
        if not (1 <= rooms <= 10):
            self.error_text.value = "Liczba pokoi musi byc między 1 a 10."
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
        """Szacuje cenę, animuje przycisk i wyniki."""

        params = self._get_params()
        if params is None:
            return

        vec = build_feature_vector(params, self.features)
        price_lr, price_rf = self.models.predict(vec)

        avg = (price_lr + price_rf) / 2
        label, color = price_label(avg)

        self.result_label.value = label
        self.result_label.color = color
        self.result_label.update()

        self.result_lr.value = f"EUR {price_lr:,.0f}"
        self.result_rf.value = f"EUR {price_rf:,.0f}"
        self.result_lr.update()
        self.result_rf.update()

        self.db.save(params, price_lr, price_rf)
        self._refresh_history()

        self._show_similar(params["area_m2"], params["rooms"])

    def _show_similar(self, area: float, rooms: int) -> None:
        """Wyświetla podobne ogłoszenia."""
        results = self.finder.find(area, rooms)
        items = [self._section("Podobne ogłoszenia")]
        for r in results:
            try:
                price = f"EUR {float(r['price_eur']):,.0f}"
            except (ValueError, KeyError):
                price = "—"
            items.append(
                ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Text(r.get("title", "")[:60], size=13, color=TEXT, weight=ft.FontWeight.W_600),
                            ft.Text(f"{r.get('area_m2', '?')} m2 · {r.get('rooms', '?')} pokoje · {price}", size=12, color=TEXT_MUTED),
                        ], spacing=2, expand=True),
                        ft.TextButton(
                            "Zobacz",
                            url=r.get("url", ""),
                            style=ft.ButtonStyle(color=GREEN),
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    bgcolor=SURFACE2,
                    border_radius=8,
                    padding=12,
                )
            )
        similar = ft.Container(
            content=ft.Column(items, spacing=8),
            bgcolor=SURFACE,
            border_radius=16,
            padding=24,
            margin=16,
            shadow=ft.BoxShadow(blur_radius=16, color="#00000018", offset=ft.Offset(0, 4)),
        )
        if len(self._main_col.controls) == 3:
            self._main_col.controls.insert(2, similar)
        else:
            self._main_col.controls[2] = similar
        self._main_col.update()

    def _refresh_history(self) -> None:
        """Odświeża tabelę historii."""
        rows = self.db.fetch_all()
        self.history_tab.rows = [
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(str(r[1])[:16], size=11, color=TEXT_MUTED)),
                ft.DataCell(ft.Text(str(r[2]), color=TEXT)),
                ft.DataCell(ft.Text(str(r[3]), color=TEXT)),
                ft.DataCell(ft.Text(LAYOUT_DISPLAY.get(r[4], str(r[4]) if r[4] else "—"), color=TEXT)),
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



def main(page: ft.Page) -> None:
    """
    Punkt wejścia aplikacji Flet.

    Args:
        page: Obiekt strony Flet.
    """
    RealEstateApp(page)


if __name__ == "__main__":
    ft.app(target=main)