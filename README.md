# Dubai Real Estate Investment Analytics

An end-to-end machine learning project that predicts Dubai property resale valuations using real transaction data — covering data cleaning, exploratory analysis, feature engineering, model training/validation, and deployment via a containerized API.

Built as a hands-on demonstration of applying data science to real estate investment analytics — combining statistical modelling with production-oriented MLOps practices (FastAPI, Docker).

---

**Live API:** https://dubai-real-estate-investment-analytics.onrender.com/docs

## Table of Contents
- [Data Source](#data-source)
- [Problem Statement](#problem-statement)
- [Exploratory Data Analysis — Key Findings](#exploratory-data-analysis--key-findings)
- [Feature Engineering](#feature-engineering)
- [Modelling Approach](#modelling-approach)
- [Results](#results)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Future Work](#future-work)

---

## Data Source

Real Dubai property transaction data from [Dubai Real Estate: Sales & Rentals (2020–2026)](https://www.kaggle.com/datasets/sergionefedov/dubai-real-estate-sales-and-rentals-20202026) on Kaggle — 87,000+ listings across 84 communities with real coordinates.

This project uses `secondary_sales.csv` (50,000 resale transactions, 2020–2026) as the core dataset. 

**Raw CSVs are not committed to this repo** (excluded via `.gitignore` — large files, redistributable from the Kaggle link above). Download the dataset and place the CSVs in a `Data/` folder at the project root before running `Train.py`.

---

## Problem Statement

Given a property's characteristics (location, size, type, age, amenities), predict its expected resale price in USD — the kind of valuation task that underpins real estate investment decisions: is an asking price reasonable, what's the expected value of a holding, how does a specific property compare to market.

---

## Exploratory Data Analysis — Key Findings

- **Price distribution is heavily right-skewed** (mean $1.01M, std $1.24M — std exceeds mean, a sign of a long tail of high-value outliers). A `log1p` transform of price produces a near-normal distribution, which is what the model is actually trained on (converted back to USD at inference time).
- **Villas sell for ~4.6x apartments' median price** ($1.4M vs. $300K) — expected given land inclusion and typical size, but confirms `property_category`/`property_type` as strong predictive features.
- **Metro distance shows a "ceiling effect," not a linear relationship.** Properties near a metro station can range from low to very high price/sqft; properties far from a metro station never reach the same ceiling. A linear model cannot capture this — it's one of the key reasons a tree-based model was chosen.
- **`area_sqft` and `bedrooms` are highly correlated (0.91)** — expected multicollinearity, handled by using a tree-based model (robust to correlated features) rather than a linear model.
- **`floor`/`total_floors` showed a spurious negative correlation with price** — traced to villas (which have no floor number) being filled with 0 and also being the higher-priced category; a confound to be aware of, not a genuine effect of floor level.
- **84 unique communities**, heavily concentrated in a subset (Emirates Hills, Bulgari Resort, Umm Suqeim, Jumeirah Bay Island lead by median price) — this cardinality and skew informed the encoding strategy below.

---

## Feature Engineering

| Step | Approach | Reasoning |
|---|---|---|
| Target transform | `log1p(price_usd)` | Corrects right-skew; predictions converted back via `expm1` at inference |
| `community` (84 categories) | **Target encoding** (mean log-price per community, computed on training data only) | One-hot would create 84 sparse columns and lose granularity if grouped; target encoding preserves per-community signal in a single numeric column |
| `property_type`, `view`, `condition`, `furnishing`, `property_category` | One-hot encoding (`drop_first=True`) | Small cardinality (≤9 categories each); `drop_first` avoids the dummy variable trap |
| `floor` / `total_floors` missing values | Filled with 0 | Missingness is fully structural — confirmed all missing values belong to villas (which don't have floor numbers), not a data quality issue |
| `area_m2` | Dropped | Perfectly redundant with `area_sqft` (same measurement, different unit) — confirmed via feature importance before/after removal showing near-identical model performance |
| `price_per_sqft_usd`, `price_per_m2_usd`, a derived `calc_price_per_sqft` sanity-check column | Dropped from features | Directly derived from the target (`price_usd / area_sqft`) — leaving these in would leak the answer into the model |
| `metro_station`, `metro_line`, `metro_distance_type` | Dropped | `metro_distance_min` (numeric) already captures the relevant signal; redundant categorical versions add no value |

**Leakage-safe target encoding:** community averages are computed strictly from the training split and applied to the test split — including a fallback (global mean) for any category unseen in training — to avoid the test set's own outcomes leaking into a feature used to predict it.

---

## Modelling Approach

- **Train/test split: time-based, not random.** Trained on transactions before Jan 1, 2025 (78.8%), tested on transactions from Jan 2025–Apr 2026 (21.2%). Real estate is time-sensitive; a random split would leak future market information into training and overstate performance. This mimics how the model would actually be used — trained on history, evaluated on the near future.
- **Baseline: Linear Regression** — established a reference point and confirmed the EDA hypothesis that price relationships here are non-linear (see Results).
- **Primary model: Random Forest Regressor** (`n_estimators=200`, `max_depth=15`) — chosen for its ability to capture non-linear relationships (metro distance ceiling effect, community pricing tiers) and interactions between features, without requiring the manual interaction terms a linear model would need.
- **Feature importance was used as a leakage sanity-check**, not just an interpretability tool — confirming `community_encoded` and `area_sqft` dominate for expected reasons (location and size are the primary real-world price drivers), with no unexplained feature dominating unexpectedly.

---

## Results

| Model | MAE (USD) | RMSE (USD) | MAPE | R² | Model Size |
|---|---|---|---|---|---|
| Linear Regression (baseline) | $242,244 | $581,289 | 17.0% | 0.956 | — |
| Random Forest (n=200, depth=15) | $148,675 | $334,006 | 11.5% | 0.981 | 282 MB |
| **Random Forest (n=100, depth=10) — deployed** | **$171,146** | — | **13.0%** | **0.976** | **14.4 MB** |

The deployed model uses fewer, shallower trees than the initial version. The larger model (282 MB) exceeded the memory limit of the free-tier deployment environment; reducing tree count and depth cut the model size by ~95% for a modest accuracy tradeoff (11.5% → 13.0% MAPE) — a deliberate deployment-efficiency decision, not an oversight.

Metrics are computed in real USD (converting predictions back from log-scale via `expm1`) rather than reported only on the log scale, since stakeholders reason in currency, not log-units.

For context, an 11.5% MAPE is broadly in line with the accuracy range reported for automated valuation models used in mature real estate markets (e.g., Zillow's Zestimate, which has historically run in a similar single-digit-to-low-teens median error range) — a reasonable result for a model built from scratch on a single transactions dataset.

**Validation against a real transaction:** a live API call for a known training-set property (DIFC, 3BR, 1,783 sqft) returned a predicted price of $2,450,423 against an actual recorded sale price of $2,287,400 — a 7.1% difference, consistent with the model's overall test-set accuracy.

---

## Deployment

- **FastAPI** service (`api/main.py`) exposing:
  - `GET /` — health check
  - `POST /predict` — accepts raw property attributes (`schemas.py` defines the expected input via Pydantic), applies the same target encoding, one-hot encoding, and column alignment used in training, and returns a predicted price in USD.
- **Dockerized** for reproducible deployment — `Dockerfile` builds a slim Python 3.12 image, installs pinned dependencies, and serves the API via `uvicorn`.
- **Dependency pinning:** `scikit-learn` is pinned to the exact version used during training, after an `InconsistentVersionWarning` surfaced a mismatch between the local training environment and a freshly-built container — a reminder that reproducibility between training and serving environments has to be enforced explicitly, not assumed.

---

## Project Structure

```
Dubai-Real-Estate-Investment-Analytics/
├── Data/                       # raw CSVs (gitignored — download from Kaggle link above)
├── Train.py                    # cleaning, feature engineering, model training & evaluation
├── models/                     # saved model artifacts (gitignored — regenerate via Train.py)
│   ├── rf_price_model.pkl
│   ├── feature_cols.pkl
│   └── community_target_map.pkl
├── api/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py               # request schema (Pydantic)
│   └── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Running Locally

```bash
pip install -r api/requirements.txt
python Train.py              # cleans data, trains model, saves artifacts to models/
cd api
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API documentation and to test the `/predict` endpoint directly.

---

## Running with Docker

```bash
docker build -t dubai-valuation-api .
docker run -p 8000:8000 dubai-valuation-api
```

---

## Design Decisions & Tradeoffs

A few decisions worth calling out explicitly, since they reflect real tradeoffs rather than defaults:

- **Target encoding over one-hot/frequency-grouping for `community`:** chosen for the stronger per-category signal it preserves at 84 categories, at the cost of needing careful leakage prevention (train-only fit) — a deliberate tradeoff of a small amount of implementation complexity for meaningfully better predictive power.
- **Random Forest over Gradient Boosting (e.g., XGBoost) for the first iteration:** prioritized robustness and simplicity to get a reliable baseline quickly; boosting is a natural next iteration once the pipeline is stable.
- **Trained model files are not committed to GitHub:** the initial Random Forest model exceeded GitHub's 100MB file size limit. Rather than introduce Git LFS under time constraints, the model is treated as a reproducible artifact (regenerate via `Train.py`) rather than a versioned file — a reasonable choice for a project at this stage, though a production setting would likely use a model registry (e.g., MLflow, S3-backed storage) instead.

---

## Future Work

- Incorporate `rentals.csv` to estimate rental yield and cap rate alongside the predicted sale price.
- Use `area_prices_monthly.csv` to build an area-level volatility/risk proxy and a simple price-momentum forecast.
- Extend into a lightweight portfolio optimization layer (mean-variance / efficient frontier across a set of properties) using the return and risk estimates above.
- Add a GenAI layer that converts the structured prediction output into a plain-English investment summary, grounded in the computed figures rather than model-generated numbers.
