# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:57:22 2026

@author: JR
"""

from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from schemas import PropertyInput

app = FastAPI(title="Dubai Real Estate Valuation API")

# Load saved artifacts once, at startup
model = joblib.load("../models/rf_price_model.pkl")
feature_cols = joblib.load("../models/feature_cols.pkl")
community_target_map = joblib.load("../models/community_target_map.pkl")
global_mean = np.mean(list(community_target_map.values))

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Dubai Real Estate Valuation API is running"}

@app.post("/predict")
def predict_price(property: PropertyInput):
    data = property.dict()

    # Target encode community (with fallback for unseen ones)
    data['community_encoded'] = community_target_map.get(data['community'], global_mean)
    del data['community']  # raw community not used directly, only its encoded version

    # Build a single-row dataframe
    df = pd.DataFrame([data])

    # One-hot encode the categorical columns, same as training
    df = pd.get_dummies(df, columns=['property_type', 'view', 'condition', 'furnishing', 'property_category'], drop_first=True)

    # Align columns to match exactly what the model was trained on
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Predict (model outputs log price, convert back to actual USD)
    log_price_pred = model.predict(df)[0]
    price_pred = np.expm1(log_price_pred)

    # Also compute a rough cap rate placeholder note - will fill with rentals data next
    return {
        "predicted_price_usd": round(float(price_pred), 2),
        "log_price": round(float(log_price_pred), 4)
    }