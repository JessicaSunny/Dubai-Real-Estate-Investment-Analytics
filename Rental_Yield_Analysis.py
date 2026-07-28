# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:43:05 2026

@author: JR
"""
#--------------------------------------------------------------------------------------------------#
import pandas as pd

sales = pd.read_csv('Data/secondary_sales.csv')
rentals = pd.read_csv('Data/rentals.csv')

print(rentals.shape)
print(rentals.dtypes)
print(rentals.isnull().sum())

pd.set_option('display.max_columns', None)
print(rentals.describe())

# 1. Metro distance outlier check
print(rentals['metro_distance_min'].sort_values(ascending=False).head(10))

# 2. Duplicates check
print(rentals.duplicated().sum())
print(rentals['id'].duplicated().sum())

# 3. Rent-per-sqft consistency check (leakage-awareness, same as secondary_sales)
rentals['calc_rent_per_sqft'] = rentals['annual_rent_usd'] / rentals['area_sqft']
print((rentals['calc_rent_per_sqft'] - rentals['rent_per_sqft_usd']).abs().describe())
rentals = rentals.drop(columns=['calc_rent_per_sqft'])


#--------------------------------------------------------------------------------------------------#
#EDA visuals
import matplotlib.pyplot as plt
import numpy as np

# 1. Rent distribution
rentals['annual_rent_usd'].hist(bins=100)
plt.title('Annual Rent Distribution')
plt.show()

# 2. Log-transformed rent
np.log1p(rentals['annual_rent_usd']).hist(bins=100)
plt.title('Log Annual Rent Distribution')
plt.show()

# 3. Rent by property category
rentals.groupby('property_category')['annual_rent_usd'].median().sort_values().plot(kind='barh')
plt.title('Median Annual Rent by Property Category')
plt.show()


# ============================================================
# RENTAL YIELD ANALYSIS
# Combines rental data with sales data to estimate expected
# rental yield / cap rate by community and property type
# ============================================================

# ------------------------------------------------------------
# Step 1: Median rent per sqft, by community and property type
# ------------------------------------------------------------
rental_summary = rentals.groupby(['community', 'property_type'])['rent_per_sqft_usd'].median().reset_index()
rental_summary.columns = ['community', 'property_type', 'median_rent_per_sqft']

print(rental_summary.head(15))
print(rental_summary.shape)

# ------------------------------------------------------------
# Step 2: Median price per sqft, by community and property type
# (from secondary_sales — reused from the price model dataset)
# ------------------------------------------------------------
price_summary = sales.groupby(['community', 'property_type'])['price_per_sqft_usd'].median().reset_index()
price_summary.columns = ['community', 'property_type', 'median_price_per_sqft']

print(price_summary.head(15))
print(price_summary.shape)

# ------------------------------------------------------------
# Step 3: Merge rent and price summaries, compute gross yield
# ------------------------------------------------------------
yield_summary = pd.merge(
    rental_summary, 
    price_summary, 
    on=['community', 'property_type'], 
    how='inner'
)

# Gross rental yield = annual rent per sqft ÷ price per sqft
yield_summary['gross_yield_pct'] = (
    yield_summary['median_rent_per_sqft'] / yield_summary['median_price_per_sqft']
) * 100

print(yield_summary.shape)
print(yield_summary.sort_values('gross_yield_pct', ascending=False).head(15))
print(yield_summary.sort_values('gross_yield_pct', ascending=True).head(15))

# ------------------------------------------------------------
# Step 4: Save yield summary for use by the API
# ------------------------------------------------------------
import os
os.makedirs('models', exist_ok=True)

yield_summary.to_csv('models/yield_summary.csv', index=False)

print("Saved: models/yield_summary.csv")
print(yield_summary.shape)