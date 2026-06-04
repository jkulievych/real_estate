# Raport Analityczny — Estymator Cen Nieruchomości w Rumunii

**Autor:** Yuliia Kuliievych  
**Data:** 2026  
**Dane:** imobiliare.ro — ogłoszenia mieszkań na sprzedaż  

---

## 1. Proces zbierania danych

Dane zebrano za pomocą scrapera napisanego w Pythonie (`scraper.py`), który pobierał ogłoszenia z portalu [imobiliare.ro](https://www.imobiliare.ro/vanzare-apartamente).

| Parametr | Wartość |
|---|---|
| Liczba stron | 333 |
| Ogłoszeń łącznie | ~9 960 |
| Ogłoszeń z podaną powierzchnią | 7 116 |
| Ogłoszeń po filtracji outlierów | 7 038 |

**Filtracja outlierów:**
- Cena: 10 000 – 2 000 000 EUR
- Powierzchnia: 10 – 500 m²
- Odrzucono 78 rekordów (0,8%)

Dane przechowywane w `data/raw/imobiliare.csv` (surowe) i `data/processed/imobiliare.csv` (przetworzone).

---

## 2. Eksploracyjna Analiza Danych (EDA)

### 2.1 Statystyki opisowe

| Zmienna | Min | Max | Średnia | Mediana | Odch. std |
|---|---|---|---|---|---|
| price_eur | 10 000 | 2 000 000 | 169 751 | 131 000 | 148 534 |
| area_m2 | 10 | 500 | 76,4 | 65,0 | 51,2 |
| rooms | 1 | 10 | 2,4 | 2 | 0,9 |
| desc_length | 0 | 5 000+ | 842 | 720 | 512 |

### 2.2 Rozkład ceny

Rozkład ceny jest **prawostronnie skośny** — większość ogłoszeń mieści się w przedziale 50 000–200 000 EUR. Rozkład logarytmiczny ceny jest znacznie bliższy normalnemu.

- Skośność: +3,2 (silna prawostronność)
- Kurtoza: 18,4 (ciężkie ogony)

### 2.3 Rozkład powierzchni

Dominują mieszkania 40–80 m². Powyżej 200 m² liczba ogłoszeń gwałtownie spada.

### 2.4 Korelacje z ceną

| Cecha | Korelacja Pearsona |
|---|---|
| area_m2 | +0.61 |
| rooms | +0.42 |
| has_pool | +0.18 |
| has_terrace | +0.15 |
| has_parking | +0.09 |
| desc_length | +0.08 |
| has_own_heating | -0.07 |

Powierzchnia jest najsilniej skorelowana z ceną. Własne ogrzewanie wykazuje słabą ujemną korelację — co sugeruje że dotyczy starszego budownictwa.

---

## 3. Transformacje zmiennych

### 3.1 Ekstrakcja cech ze słownika (features.py)

Z opisów ogłoszeń wyekstrahowano 10 zmiennych binarnych metodą słownikową (regex):

| Cecha | Wzorzec | Częstość |
|---|---|---|
| has_parking | `parcar` | 47% |
| has_balcony | `balcon` | 46% |
| has_own_heating | `centrală proprie\|centrală termic` | 42% |
| has_metro | `metrou\|metro` | 26% |
| has_ac | `aer condiționat\|climatiz` | 24% |
| has_elevator | `lift\|ascensor` | 22% |
| has_terrace | `terasă` | 20% |
| has_new_building | `bloc nou\|construcție nou` | 7% |
| has_garden | `grădină` | 5% |
| has_pool | `piscin` | 4% |

Dodatkowa cecha inżynieryjna: `desc_length` — długość opisu w znakach.

### 3.2 Kodowanie wskaźnikowe (one-hot encoding)

Zmienna `layout_type` (typ układu) zakodowana jako 5 zmiennych binarnych z kategorią odniesienia `decomandat` (rozdzielny):

| Kolumna | Liczebność |
|---|---|
| layout_semidecomandat | 2 175 |
| layout_nedefinit | 374 |
| layout_nedecomandat | 103 |
| layout_garsoniera | 76 |
| layout_penthouse | 8 |

---

## 4. Historia budowy modelu regresji

### 4.1 Podział danych

- Zbiór treningowy: 5 630 rekordów (80%)
- Zbiór testowy: 1 408 rekordów (20%)
- Podział losowy z ziarnem 42

### 4.2 Cechy wejściowe (18 zmiennych)

```
area_m2, rooms,
layout_garsoniera, layout_nedecomandat, layout_nedefinit,
layout_penthouse, layout_semidecomandat,
has_parking, has_balcony, has_elevator, has_pool,
has_metro, has_ac, has_terrace, has_garden,
has_new_building, has_own_heating, desc_length
```

### 4.3 Regresja liniowa (OLS)

Implementacja własna metodą najmniejszych kwadratów bez zewnętrznych bibliotek ML.

**Wyniki na zbiorze testowym:**

| Metryka | Wartość |
|---|---|
| R² | 0.4447 |
| RMSE | 115 409 EUR |
| MAE | 63 852 EUR |

**Najważniejsze współczynniki:**

| Cecha | Współczynnik |
|---|---|
| has_terrace | +55 574 |
| has_pool | +50 958 |
| rooms | +50 000 |
| layout_garsoniera | +36 729 |
| has_parking | +23 129 |
| has_own_heating | -18 757 |
| layout_nedecomandat | -14 856 |

### 4.4 Las losowy (Random Forest)

100 drzew decyzyjnych, sklearn `RandomForestRegressor`.

**Wyniki na zbiorze testowym:**

| Metryka | Wartość |
|---|---|
| R² | 0.5626 |
| RMSE | 102 423 EUR |
| MAE | 50 754 EUR |

### 4.5 Diagnostyka modeli

**Problemy regresji liniowej:**
- R² = 0.44 — model wyjaśnia tylko 44% wariancji ceny
- Wysoki RMSE sugeruje duże błędy przy przewidywaniu bardzo drogich nieruchomości
- Brak zmiennych lokalizacyjnych (koordynaty GPS, dzielnica) znacząco ogranicza jakość

**Las losowy lepszy o:**
- R²: +0.118 (+26%)
- RMSE: -12 986 EUR (-11%)
- MAE: -13 098 EUR (-21%)

---

## 5. Końcowy model użyty w aplikacji

W aplikacji GUI (`app.py`) używane są **oba modele równolegle**:
- Regresja liniowa → wynik LR
- Las losowy → wynik RF
- Ocena ceny bazuje na średniej obu modeli

**Granice oceny ceny:**
- Okazja: < 80 000 EUR
- Przeciętna: 80 000 – 150 000 EUR
- Wysoka: > 150 000 EUR

---

## 6. Ograniczenia modelu

1. **Brak danych lokalizacyjnych** — brak koordynat GPS ani kodu pocztowego znacząco obniża R²
2. **Subiektywność opisów** — cechy wyekstrahowane ze słownika zależą od tego co sprzedający napisał w opisie
3. **Dane tylko z Rumunii** — model nie nadaje się do innych krajów
4. **Outlierzy cenowi** — nieruchomości > 500 000 EUR są słabo reprezentowane w danych

---

## 7. Możliwe usprawnienia

- Dodanie zmiennych lokalizacyjnych (geocoding adresu)
- Zastosowanie transformacji logarytmicznej ceny jako zmiennej docelowej
- Większy zbiór danych (pełne 70 000 ogłoszeń)
- Cross-validacja zamiast pojedynczego podziału train/test
- Gradient Boosting (XGBoost, LightGBM) jako alternatywa dla RF
