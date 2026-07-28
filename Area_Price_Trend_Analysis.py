# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:44:17 2026

@author: JR
"""
#--------------------------------------------------------------------------------------------------#

import pandas as pd

area_prices = pd.read_csv('Data/area_prices_monthly.csv')

print(area_prices.shape)
print(area_prices.dtypes)
print(area_prices.isnull().sum())

pd.set_option('display.max_columns', None)
print(area_prices.describe())
print(area_prices.head(10))

#--------------------------------------------------------------------------------------------------#

# Confirm the off-plan missingness theory (some communities may just never have off-plan listings)
missing_offplan_communities = area_prices[area_prices['offplan_price_per_sqft_usd'].isnull()]['community'].nunique()
print(f"Communities with missing offplan data: {missing_offplan_communities}")

# Confirm full date coverage per community (should be 76 months each, if complete)
print(area_prices.groupby('community')['year_month'].count().describe())

# Duplicates check
print(area_prices.duplicated().sum())

#--------------------------------------------------------------------------------------------------#

# ------------------------------------------------------------
# Step 1: Prepare data - ensure proper date type and sort order
# ------------------------------------------------------------
area_prices['year_month'] = pd.to_datetime(area_prices['year_month'])
area_prices = area_prices.sort_values(['community', 'year_month'])

# ------------------------------------------------------------
# Step 2: Compute month-over-month % change in price, per community
# ------------------------------------------------------------
area_prices['pct_change'] = area_prices.groupby('community')['secondary_price_per_sqft_usd'].pct_change()

# ------------------------------------------------------------
# Step 3: Aggregate into volatility (risk) and trend (momentum) per community
# ------------------------------------------------------------
risk_trend_summary = area_prices.groupby('community').agg(
    volatility_pct=('pct_change', 'std'),          # risk proxy: how much price swings month to month
    avg_monthly_growth_pct=('pct_change', 'mean'),  # trend/momentum: average growth direction
    latest_price_per_sqft=('secondary_price_per_sqft_usd', 'last'),
    price_2020=('secondary_price_per_sqft_usd', 'first'),
).reset_index()

# Total growth over the full period, for context
risk_trend_summary['total_growth_pct'] = (
    (risk_trend_summary['latest_price_per_sqft'] / risk_trend_summary['price_2020']) - 1
) * 100

print(risk_trend_summary.sort_values('volatility_pct', ascending=False).head(10))
print(risk_trend_summary.sort_values('total_growth_pct', ascending=False).head(10))

#--------------------------------------------------------------------------------------------------#
risk_trend_summary.to_csv('models/risk_trend_summary.csv', index=False)
print("Saved: models/risk_trend_summary.csv")