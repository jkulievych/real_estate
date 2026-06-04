# Estymator Cen Nieruchomości

Aplikacja do szacowania cen mieszkań na podstawie danych z rumuńskiego portalu [imobiliare.ro](https://www.imobiliare.ro).

## Opis projektu

Projekt składa się z czterech głównych modułów:

- **scraper.py** — pobiera ogłoszenia nieruchomości z imobiliare.ro
- **features.py** — wyciąga dodatkowe cechy z opisów ogłoszeń (parking, balkon, metro itp.)
- **model.py** — trenuje modele regresji liniowej i lasu losowego do przewidywania cen
- **app.py** — aplikacja GUI (Flet) do szacowania cen przez użytkownika

## Dane

Zebrano ponad 7000 ogłoszeń mieszkań na sprzedaż w Rumunii. Po filtracji (tylko rekordy z podaną powierzchnią, ceny 10 000–2 000 000 EUR) zbiór liczy ~7000 rekordów.

## Modele

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Regresja liniowa | 0.44 | 115 408 EUR | 63 852 EUR |
| Las losowy | 0.56 | 102 423 EUR | 50 753 EUR |

## Instalacja

```bash
pip install -r requirements.txt
python features.py   # ekstrakcja cech
python model.py      # trening modeli
python app.py        # uruchomienie aplikacji
```

## Testy

```bash
python -m pytest tests/ -v
```

## Autor

Yuliia Kuliievych — projekt zaliczeniowy

## Licencja

MIT
