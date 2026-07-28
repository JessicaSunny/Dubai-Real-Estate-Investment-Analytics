# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:57:21 2026

@author: JR
"""
#--------------------------------------------------------------------------------------------------#
# ============================================================
# PORTFOLIO OPTIMIZATION
# Uses expected return (price momentum) and risk (volatility +
# correlation) from area_prices_monthly to find optimal
# community weightings for a hypothetical investment portfolio
# ============================================================

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
area_prices = pd.read_csv('Data/area_prices_monthly.csv')
risk_trend_summary = pd.read_csv('models/risk_trend_summary.csv')

# ------------------------------------------------------------
# Step 1: Select communities to consider for the portfolio
# ------------------------------------------------------------
selected_communities = [
    'Remraam', 'Arabian Ranches 2', 'Dubai South', 'Green Community',
    'The Meadows', 'Palm Jumeirah', 'DIFC', 'Business Bay'
]

portfolio_data = risk_trend_summary[risk_trend_summary['community'].isin(selected_communities)].copy()
portfolio_data = portfolio_data.set_index('community')

print(portfolio_data[['avg_monthly_growth_pct', 'volatility_pct']])

# ------------------------------------------------------------
# Step 2: Reshape area_prices to wide format
# Rows = months, Columns = communities, Values = price per sqft
# ------------------------------------------------------------
price_wide = area_prices.pivot(
    index='year_month',
    columns='community',
    values='secondary_price_per_sqft_usd'
)

print(price_wide.shape)
print(price_wide.head())

# ------------------------------------------------------------
# Step 3: Convert to month-over-month % change
# ------------------------------------------------------------
returns_wide = price_wide.pct_change().dropna()

print(returns_wide.shape)
print(returns_wide.head())