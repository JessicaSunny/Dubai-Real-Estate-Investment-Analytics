# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:08:32 2026

@author: JR
"""
#--------------------------------------------------------------------------------------------------#
import pandas as pd

sales = pd.read_csv('secondary_sales.csv')

print(sales.shape)
print(sales.dtypes)
print(sales.isnull().sum())
print(sales.describe())

# Confirm the floor/total_floors missingness theory
print(sales[sales['floor'].isnull()]['property_category'].value_counts())

#Get describe on the specific columns that matter, so nothing's hidden
pd.set_option('display.max_columns', None)
print(sales[['price_usd', 'area_sqft', 'bedrooms', 'year_built', 
             'price_per_sqft_usd', 'metro_distance_min']].describe())

#Decision on floor/total_floors
sales['floor'] = sales['floor'].fillna(0)
sales['total_floors'] = sales['total_floors'].fillna(0)

# duplicates
print(sales.duplicated().sum())
print(sales['id'].duplicated().sum())

# sanity check: any price_per_sqft that doesn't match price/area (data entry errors)
sales['calc_price_per_sqft'] = sales['price_usd'] / sales['area_sqft']
print((sales['calc_price_per_sqft'] - sales['price_per_sqft_usd']).abs().describe())
#A max difference of $3.67 on prices per sqft that range into the hundreds is noise, not a data quality problem, hence dropping it
sales = sales.drop(columns=['calc_price_per_sqft'])

#--------------------------------------------------------------------------------------------------#
#EDA visuals
import matplotlib.pyplot as plt

# 1. Price distribution — check the skew we flagged earlier
sales['price_usd'].hist(bins=100)
plt.title('Price Distribution')
plt.show()

# 2. Log-transformed price — see if this looks more "normal"
import numpy as np
np.log1p(sales['price_usd']).hist(bins=100)
plt.title('Log Price Distribution')
plt.show()

# 3. Price by property category
sales.groupby('property_category')['price_usd'].median().sort_values().plot(kind='barh')
plt.title('Median Price by Property Category')
plt.show()

#Deeper Analysis
# 4. Price by community — which areas command a premium?
sales.groupby('community')['price_usd'].median().sort_values(ascending=False).head(15).plot(kind='barh')
plt.title('Top 15 Communities by Median Price')
plt.gca().invert_yaxis()
plt.show()

# 5. Does metro distance actually relate to price?
plt.scatter(sales['metro_distance_min'], sales['price_per_sqft_usd'], alpha=0.1)
plt.xlabel('Metro Distance (min)')
plt.ylabel('Price per sqft (USD)')
plt.title('Metro Distance vs Price per sqft')
plt.show()

# 6. Correlation heatmap on numeric features
import seaborn as sns
numeric_cols = ['price_usd', 'area_sqft', 'bedrooms', 'floor', 'total_floors', 
                 'year_built', 'metro_distance_min', 'to_burj_khalifa_km', 'mortgage_rate_at_listing']
sns.heatmap(sales[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.show()

# How many unique communities/property types are we dealing with?
print(sales['community'].nunique())
print(sales['property_category'].value_counts())
print(sales['property_type'].value_counts())
print(sales['view'].nunique())
print(sales['condition'].value_counts())
print(sales['furnishing'].value_counts())

#--------------------------------------------------------------------------------------------------#
#ENCODING
# Check if zone is just a coarser version of community (likely redundant)
print(sales.groupby('zone')['community'].nunique().sort_values(ascending=False).head(10))

# Confirm date range for our eventual time-based split
sales['date_listed'] = pd.to_datetime(sales['date_listed'])
print(sales['date_listed'].min(), sales['date_listed'].max())

#cutoff and split
cutoff_date = '2025-01-01'

train = sales[sales['date_listed'] < cutoff_date].copy()
test = sales[sales['date_listed'] >= cutoff_date].copy()

print(f"Train: {train.shape[0]} rows ({train.shape[0]/len(sales)*100:.1f}%)")
print(f"Test: {test.shape[0]} rows ({test.shape[0]/len(sales)*100:.1f}%)")
print(f"Train date range: {train['date_listed'].min()} to {train['date_listed'].max()}")
print(f"Test date range: {test['date_listed'].min()} to {test['date_listed'].max()}")

#target encoding on top of this split
import numpy as np

# Log-transform the target (recall: price is right-skewed)
train['log_price'] = np.log1p(train['price_usd'])
test['log_price'] = np.log1p(test['price_usd'])

# Compute community averages using TRAIN ONLY
community_target_map = train.groupby('community')['log_price'].mean()

# Apply to both train and test
train['community_encoded'] = train['community'].map(community_target_map)
test['community_encoded'] = test['community'].map(community_target_map)

# Handle any community in test that never appeared in train
global_mean = train['log_price'].mean()
missing_before_fill = test['community_encoded'].isnull().sum()
test['community_encoded'] = test['community_encoded'].fillna(global_mean)

print(f"Communities in train: {train['community'].nunique()}")
print(f"Communities in test: {test['community'].nunique()}")
print(f"Test rows needing fallback (unseen community): {missing_before_fill}")
print(train[['community', 'community_encoded']].drop_duplicates().head(10))

#encode the remaining categoricals
categorical_cols = ['property_type', 'view', 'condition', 'furnishing']

# One-hot encode both, then align columns (handles any category mismatch between train/test)
train_encoded = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# Align: make sure test has exactly the same columns as train (fill missing with 0)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

print(train_encoded.shape)
print(test_encoded.shape)