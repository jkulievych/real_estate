# 🏠 Romanian Real Estate Price Estimator

> *How much is that apartment in Cluj really worth? I scraped 7,116 listings to find out.*

![Demo](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGh4bHRicGYyNnZuMzFpNXVwZThoNXd1ZjRqendqMGV6cHp6bTZ3aCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LaVp0AyqR5bGsC5Cbm/giphy.gif)
![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXB2MDN4MXBhaTF6OWh5YmFhNzJ2aXIxaHgyNGxxdWQyejhscGY5cSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/nnkG9QTmPVcLgMZUV9/giphy.gif)
---

## What This Is

A full-stack machine learning project that estimates apartment prices in Romania using data scraped from [imobiliare.ro](https://imobiliare.ro) — the country's largest real estate portal.

You enter basic apartment details. The app instantly returns a **price estimate with a confidence interval**, comparable listings, and a breakdown of which features matter most. Everything runs locally — no API keys, no subscriptions.

---

## Features

| | |
|---|---|
| 🕷️ **Web scraper** | Collects listings via JSON-LD structured data — fast and robust against layout changes |
| 🔬 **EDA notebook** | Exploratory analysis of 7,116 listings across Romanian cities |
| 🤖 **Two models** | Custom OLS (plain Python, no sklearn) + Random Forest (sklearn) |
| 📊 **Confidence intervals** | Bootstrap-based uncertainty estimates on every prediction |
| 🖥️ **GUI app** | Flet desktop app with Lottie animations and prediction history |
| ✅ **69 tests** | pytest suite covering scraper, feature extraction, and model logic |
| 📚 **MkDocs docs** | Full API documentation with pdoc |

---

## Dataset

- **Source:** imobiliare.ro (scraped May 2025)
- **Listings:** 7,116 apartment offers
- **Coverage:** Bucharest, Cluj-Napoca, Timișoara, Brașov, Iași, and 15+ smaller cities
- **Features:** 18 engineered features per listing

Key features used by the model:

| Feature | Description |
|---|---|
| `area_m2` | Total usable area in square metres |
| `rooms` | Number of rooms |
| `floor` | Floor number (0 = ground floor) |
| `total_floors` | Total floors in building |
| `city` | City (one-hot encoded) |
| `layout_type` | Layout: decomandat / semidecomandat / garsonieră / studio |
| `year_built` | Construction year |
| `has_parking` | Parking space included |
| `has_balcony` | Balcony present |
| `has_storage` | Storage room included |

---

## Model Performance

| Model | MAE (EUR) | RMSE (EUR) | R² |
|---|---|---|---|
| OLS (custom) | ~63,400 | ~115,100 | 0.44 |
| Random Forest | ~50,800 | ~102,400 | 0.56 |

Random Forest is the default predictor in the app. OLS is included for interpretability and coursework requirements.

---

## Project Structure

```
imobiliare-estimator/
│
├── scraper.py              # imobiliare.ro scraper (JSON-LD)
├── features.py             # Feature extraction from raw listing dicts
├── model.py                # OLS (custom) + Random Forest wrapper
├── app.py                  # Flet GUI application
│
├── data/
│   ├── raw/                # Raw scraped JSON files
│   └── processed/          # Cleaned, feature-engineered CSVs
│
├── models/
│   ├── rf_model.pkl        # Trained Random Forest
│   └── ols_model.pkl       # Trained OLS model
│
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
│
├── tests/                  # pytest test suite (69 tests)
│   ├── test_scraper.py
│   ├── test_features.py
│   └── test_model.py
│
├── docs/                   # MkDocs documentation
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/jkulievych/real_estate.git
cd imobiliare-estimator

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install flet==0.85.0.dev1
pip install flet-lottie
```

---

## Usage

### Run the GUI app

```bash
python app.py
```

The app opens a desktop window. Enter apartment details, click **Estimate**, and get an instant price prediction with confidence interval.

### Scrape fresh data

```bash
python scraper.py --city cluj-napoca --pages 50
```

Data is saved to `data/raw/` as JSON files.

### Retrain models

```bash
python model.py --train --data data/processed/listings.csv
```

Trained models are saved to `models/`.

### Run the EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

### Run tests

```bash
pytest tests/ -v
```

### Build documentation

```bash
mkdocs serve          # local preview at http://127.0.0.1:8000
mkdocs build          # static HTML to site/
```

---



## How It Works — The Short Version

1. **Scraping:** The scraper fetches listing pages and parses `application/ld+json` script blocks embedded in the HTML. This is more stable than CSS selectors since it targets structured data that imobiliare.ro publishes for search engines.

2. **Feature engineering:** Raw listing dicts go through `features.py`, which uses regex and lookup tables to extract numeric and categorical features. Layout type labels are standardised to Romanian real-estate vocabulary (*decomandat*, *semidecomandat*, etc.).

3. **Modelling:** Two models are trained:
   - **OLS** — implemented from scratch (no sklearn) using the normal equation `β = (XᵀX)⁻¹Xᵀy`. Included for coursework and interpretability.
   - **Random Forest** — sklearn's `RandomForestRegressor`, tuned with cross-validation. This is the production predictor.
   
4. **Confidence intervals** are computed via bootstrap resampling (1,000 samples) at the prediction stage.

5. **GUI:** Built with [Flet](https://flet.dev), which compiles Python to a Flutter desktop app. Prediction history is persisted in a local SQLite database.


---

## Requirements

```
requests
beautifulsoup4
lxml
scikit-learn>=1.4
numpy
pandas
matplotlib
seaborn
flet==0.85.0.dev1
flet-lottie
pytest
mkdocs
mkdocs-material
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

Built as a coursework project (Extraction and Analysis of Unstructured Data; Fundamentals of Programming in Python II). 
Data sourced from imobiliare.ro for educational purposes only.
![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGMzZWh4c3JvbGswdGptOXJ2N3dicm10Y3M2dDA4cmtmbzl1dWI2dyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/13ETTwaYwKMfmg/giphy.gif)
