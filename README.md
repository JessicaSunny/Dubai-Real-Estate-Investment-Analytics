# Dubai Real Estate Investment Analytics

An end-to-end investment analytics project built on real Dubai transaction data — covering property valuation (ML), rental yield analysis, market trend/risk profiling, and portfolio optimization. Includes a deployed prediction API and an interactive dashboard combining all three analyses.

**🔗 Live API (FastAPI, price prediction):** https://dubai-real-estate-investment-analytics.onrender.com/docs
**🔗 Live Dashboard (Streamlit, full analysis):** https://dubai-real-estate-investment-analytics-jyvtswntqky3ephmaagnrh.streamlit.app/

*Note: the Render API free tier spins down after inactivity — the first request may take 30–60 seconds to respond.*

---

## Table of Contents
- [Data Source](#data-source)
- [Project Components](#project-components)
- [1. Property Valuation Model](#1-property-valuation-model)
- [2. Rental Yield Analysis](#2-rental-yield-analysis)
- [3. Market Trend & Risk Analysis](#3-market-trend--risk-analysis)
- [4. Portfolio Optimization](#4-portfolio-optimization)
- [Interactive Dashboard](#interactive-dashboard)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)

---

## Data Source

Real Dubai property data from [Dubai Real Estate: Sales & Rentals (2020–2026)](https://www.kaggle.com/datasets/sergionefedov/dubai-real-estate-sales-and-rentals-20202026) on Kaggle (Apache 2.0 licensed) — 87,000+ listings across 84 communities with real coordinates.

Three of the five source files are used in this project:
- `secondary_sales.csv` (50,000 rows) — resale transactions, used for the price model
- `rentals.csv` (25,000 rows) — rental listings, used for yield analysis
- `area_prices_monthly.csv` (6,384 rows) — monthly price indices by community (2020–2026), used for risk/trend analysis and portfolio optimization

`metro_stations.csv` and `off_plan.csv` are part of the source dataset but not used — metro proximity is already captured numerically in the other files, and off-plan pricing follows different (developer/payment-plan-driven) dynamics outside this project's scope.

---

## Project Components

| Component | Type | Script |
|---|---|---|
| Property valuation | Trained ML model (Random Forest) | `Train.py` |
| Rental yield | Statistical/KPI aggregation | `Rental_Yield_Analysis.py` |
| Market trend & risk | Statistical/KPI aggregation | `Area_Price_Trend_Analysis.py` |
| Portfolio optimization | Mean-variance optimization (`scipy`) | `Portfolio_Optimization.py` |

Only the price valuation component is a trained predictive model deployed via API; the other three are calculation-based analyses surfaced through the dashboard. This is a deliberate distinction — not every problem benefits from a trained model, and the yield/trend/optimization pieces are well-served by direct statistical computation rather than an unnecessary ML layer.

---

## 1. Property Valuation Model

**Goal:** predict resale price in USD from property characteristics.

**Key EDA findings:**
- Price is heavily right-skewed (log1p transform applied to the target)
- Villas sell for ~4.6x apartments' median price
- Metro distance shows a "ceiling effect" — a non-linear relationship with price/sqft, not a straight-line relationship
- `area_sqft` and `bedrooms` are highly correlated (0.91) — handled via a tree-based model rather than linear regression

**Feature engineering:**
- Target encoding for `community` (84 categories), fit on the training split only to prevent leakage
- One-hot encoding for smaller categorical fields (`property_type`, `view`, `condition`, `furnishing`, `property_category`)
- Removed direct target-derived features (`price_per_sqft_usd`, `price_per_m2_usd`) to prevent leakage
- Removed `area_m2` (perfectly redundant with `area_sqft`)

**Validation:** time-based train/test split (train ≤ 2024, test 2025–2026) to avoid leaking future market information.

**Results:**

| Model | MAE (USD) | RMSE (USD) | MAPE | R² | Model Size |
|---|---|---|---|---|---|
| Linear Regression (baseline) | $242,244 | $581,289 | 17.0% | 0.956 | — |
| Random Forest (n=200, depth=15) — initial | $148,675 | $334,006 | 11.5% | 0.981 | 282 MB |
| **Random Forest (n=100, depth=10) — deployed** | **$171,146** | **$402,555** | **13.0%** | **0.976** | **14.4 MB** |

The deployed model uses fewer, shallower trees than the version initially trained. The larger model exceeded the memory limit of the free-tier deployment environment (Render, 512MB); reducing tree count and depth cut model size by ~95% for a modest accuracy tradeoff — a deliberate deployment-efficiency decision. Only the deployed configuration is in the current codebase.

**Validation against a real transaction:** a live API call for a known property (DIFC, 3BR, 1,783 sqft) returned $1,998,777 against an actual recorded sale price of $2,287,400 — a 12.6% difference, consistent with the deployed model's overall MAPE.

---

## 2. Rental Yield Analysis

**Goal:** estimate expected gross rental yield by community and property type.

**Method:** median rent-per-sqft (from `rentals.csv`) divided by median sale-price-per-sqft (from `secondary_sales.csv`), aggregated by community + property type, expressed as a percentage. This is **gross** yield (rent as a % of price, before operating expenses) — no separate cap rate calculation was made, since operating cost data wasn't available in the source dataset.

**Key finding:** clear yield compression in prime locations — affordable, family-oriented communities (Dubai South, The Valley, International City) cluster around 10–12% gross yield, while ultra-prime addresses (Bulgari Resort, DIFC, Downtown Dubai, Pearl Jumeira) compress to 5–6%. This reflects a well-known real estate pattern: prime-area pricing is driven more by prestige than by rental income potential.

Output: `models/yield_summary.csv` (505 community × property-type combinations).

---

## 3. Market Trend & Risk Analysis

**Goal:** quantify historical price growth and volatility (as a risk proxy) by community, using 76 months of data per community (Jan 2020–Apr 2026).

**Method:**
- Month-over-month % change in `secondary_price_per_sqft_usd`, per community
- **Volatility** = standard deviation of monthly % change (risk proxy)
- **Average monthly growth** and **total growth (2020→2026)** as return/momentum indicators

**Key finding:** the highest-volatility communities (Al Warqa, Arabian Ranches 2, Remraam, Dubai South) substantially overlap with the highest-total-growth communities — a direct, data-driven illustration of the risk-return relationship: communities that appreciated most over six years also swung the most month to month.

Output: `models/risk_trend_summary.csv` (84 communities).

---

## 4. Portfolio Optimization

**Goal:** given a set of communities, find the allocation that maximizes risk-adjusted return — the direct analytical answer to "support portfolio optimization to maximize ROI and capital efficiency."

**Method (mean-variance optimization):**
1. Reshaped monthly price data into wide format (months × communities) and computed the **correlation matrix** of monthly returns across 8 selected communities
2. Built a full **covariance matrix** (`correlation × volatility_i × volatility_j`) rather than treating each community's risk in isolation
3. Used `scipy.optimize` to maximize a Sharpe-style ratio (`return / risk`) subject to weights summing to 1, no short-selling

**Why the covariance matrix mattered:** an earlier version of this analysis used only average individual volatility (no correlation data) as the risk measure. That version mathematically collapsed to a 100% allocation in a single community — a symptom of the missing ingredient, not a real result, since with no correlation structure there is no mathematical benefit to diversifying. Incorporating the real correlation matrix (several selected community pairs showed correlations from -0.22 to +0.40 — meaningfully independent or even inversely related movement) produced a genuinely diversified allocation across all 8 communities.

**Result:**

| | Return (monthly) | Risk (volatility) |
|---|---|---|
| Equal-weighted portfolio | 0.860% | 1.817% |
| **Optimized portfolio** | **0.887%** | **1.727%** |

The optimized portfolio achieves both higher return and lower risk than an equal-weighted split — the core demonstrable benefit of diversification, achieved specifically by weighting more heavily into communities with low or negative correlation to one another.

**Known simplification:** the analysis is scoped to 8 selected communities rather than optimizing across all 84, given time constraints — a full production version would extend the same method to the complete set.

Output: `models/portfolio_optimization_result.csv`.

---

## Interactive Dashboard

A Streamlit app (`streamlit_app.py`) unifies all three analyses into one interface:

- **Tab 1 — Property Valuation:** enter property details, get a predicted price (from the deployed model) and expected gross rental yield (from the yield analysis) in one view
- **Tab 2 — Market Trends & Risk:** select a community, view its historical growth/volatility profile plus city-wide comparison charts
- **Tab 3 — Portfolio Optimization:** view the optimized allocation, and the equal-weighted vs. optimized comparison

The dashboard loads the trained model and analysis outputs directly from the repository rather than calling the deployed FastAPI service — a deliberate simplicity/reliability tradeoff for a live demo context. A production version would have the dashboard call the API instead, separating the presentation layer from model-serving.

---

## Deployment

- **FastAPI + Docker**, deployed on Render: exposes `/predict` for the price model
  - Dependency version pinned (`scikit-learn`) after an environment mismatch surfaced between local training and the container build
  - Model retrained during the Docker build step (`RUN python Train.py`) rather than copying a pre-built artifact, since the initial 282MB model exceeded GitHub's file size limit; the deployed model was subsequently reduced to 14.4MB, small enough to commit directly
- **Streamlit Community Cloud**: hosts the interactive dashboard directly from this GitHub repository

---

## Project Structure

```
Dubai-Real-Estate-Investment-Analytics/
├── Data/                                # raw CSVs (Apache 2.0 licensed, redistributed here)
├── Train.py                             # price model: cleaning, features, training, evaluation
├── Rental_Yield_Analysis.py             # yield calculation
├── Area_Price_Trend_Analysis.py         # volatility/trend/growth calculation
├── Portfolio_Optimization.py            # correlation matrix + mean-variance optimization
├── streamlit_app.py                     # unified interactive dashboard
├── requirements.txt                     # root-level, for Streamlit Cloud
├── models/
│   ├── rf_price_model.pkl
│   ├── feature_cols.pkl
│   ├── community_target_map.pkl
│   ├── yield_summary.csv
│   ├── risk_trend_summary.csv
│   └── portfolio_optimization_result.csv
├── api/
│   ├── main.py                          # FastAPI app (price prediction endpoint)
│   ├── schemas.py
│   └── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Running Locally

**Dashboard:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**API:**
```bash
pip install -r api/requirements.txt
python Train.py              # regenerates model artifacts if needed
cd api && uvicorn main:app --reload
```

**Docker (API only):**
```bash
docker build -t dubai-valuation-api .
docker run -p 8000:8000 dubai-valuation-api
```

---

## Design Decisions & Tradeoffs

- **Target encoding over one-hot/frequency-grouping for `community`:** stronger per-category signal at 84 categories, at the cost of needing careful leakage prevention (train-only fit).
- **Random Forest sized down for deployment:** a ~95% model size reduction (282MB → 14.4MB) for a 1.5-point MAPE tradeoff (11.5% → 13.0%) — a deliberate production constraint, not an oversight.
- **Gross yield, not net cap rate:** operating expense data wasn't available in the source dataset; gross yield is reported explicitly as such rather than presented as a full cap rate.
- **Portfolio optimization scoped to 8 communities, with a full covariance matrix:** rather than a larger but oversimplified version. An earlier attempt using only individual volatility (no correlation) produced a degenerate, single-asset-concentrated result — corrected by computing the real correlation structure, which is the actual mechanism that makes diversification mathematically meaningful.
- **Dashboard loads models directly rather than calling the deployed API:** prioritizes demo reliability (no dependency on Render's cold-start behavior) over architectural purity; noted explicitly as a simplification rather than presented as the production-ideal design.
