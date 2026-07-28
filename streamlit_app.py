# ============================================================
# DUBAI REAL ESTATE INVESTMENT ANALYTICS — STREAMLIT APP
# Combines: price prediction, rental yield, and portfolio
# optimization into one interactive dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dubai Real Estate Investment Analytics", layout="wide")

# ------------------------------------------------------------
# Load all saved artifacts once, at startup
# ------------------------------------------------------------
model = joblib.load("models/rf_price_model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
community_target_map = joblib.load("models/community_target_map.pkl")
yield_summary = pd.read_csv("models/yield_summary.csv")
risk_trend_summary = pd.read_csv("models/risk_trend_summary.csv")
portfolio_result = pd.read_csv("models/portfolio_optimization_result.csv")

st.title("🏙️ Dubai Real Estate Investment Analytics")
st.caption("Price valuation, rental yield, and portfolio optimization — built on real Dubai transaction data (2020–2026)")

tab1, tab2, tab3 = st.tabs(["🏠 Property Valuation", "📈 Market Trends & Risk", "📊 Portfolio Optimization"])

# ============================================================
# TAB 1: PROPERTY VALUATION
# ============================================================
with tab1:
    st.header("Property Valuation & Expected Yield")
    st.write("Enter property details to get an estimated price and expected rental yield.")

    col1, col2, col3 = st.columns(3)

    with col1:
        community = st.selectbox("Community", sorted(community_target_map.index))
        property_type = st.selectbox("Property Type", 
            ['1BR', '2BR', '3BR', 'studio', '3BR_villa', '4BR_penthouse', '4BR_villa', '5BR_villa', '6BR_villa'])
        property_category = st.selectbox("Property Category", ['apartment', 'villa'])

    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=6, value=3)
        area_sqft = st.number_input("Area (sqft)", min_value=300, max_value=13000, value=1500)
        floor = st.number_input("Floor", min_value=0, max_value=80, value=10)
        total_floors = st.number_input("Total Floors", min_value=0, max_value=90, value=30)

    with col3:
        view = st.selectbox("View", ['city', 'community', 'golf_course', 'marina', 'park', 'pool', 'sea', 'burj_khalifa'])
        condition = st.selectbox("Condition", ['vacant_on_transfer', 'tenanted', 'off_plan_resale'])
        furnishing = st.selectbox("Furnishing", ['unfurnished', 'fully_furnished', 'semi_furnished'])

    metro_distance_min = st.slider("Distance to Metro (minutes)", 0, 200, 15)
    to_burj_khalifa_km = st.slider("Distance to Burj Khalifa (km)", 0.0, 40.0, 10.0)

    if st.button("Get Valuation", type="primary"):
        input_data = {
            'is_freehold': True,
            'lat': 25.10,
            'lon': 55.25,
            'bedrooms': bedrooms,
            'area_sqft': area_sqft,
            'floor': floor,
            'total_floors': total_floors,
            'year_built': 2020,
            'parking_spaces': 1,
            'chiller_included': True,
            'metro_distance_min': metro_distance_min,
            'to_burj_khalifa_km': to_burj_khalifa_km,
            'mortgage_rate_at_listing': 5.0,
            'community_encoded': community_target_map.get(community, np.mean(list(community_target_map.values))),
        }

        df = pd.DataFrame([input_data])
        df = pd.get_dummies(pd.concat([df, pd.DataFrame([{
            'property_type': property_type, 'view': view, 'condition': condition,
            'furnishing': furnishing, 'property_category': property_category
        }])], axis=1), columns=['property_type', 'view', 'condition', 'furnishing', 'property_category'], drop_first=True)

        df = df.reindex(columns=feature_cols, fill_value=0)

        log_price_pred = model.predict(df)[0]
        price_pred = np.expm1(log_price_pred)

        # Look up expected yield for this community + property type
        yield_row = yield_summary[
            (yield_summary['community'] == community) & 
            (yield_summary['property_type'] == property_type)
        ]
        expected_yield = yield_row['gross_yield_pct'].values[0] if len(yield_row) > 0 else None

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Estimated Price", f"${price_pred:,.0f}")
        with col_b:
            if expected_yield is not None:
                st.metric("Expected Gross Rental Yield", f"{expected_yield:.2f}%")
                st.metric("Estimated Annual Rent", f"${price_pred * expected_yield / 100:,.0f}")
            else:
                st.warning("No yield data available for this exact community + property type combination.")

# ============================================================
# TAB 2: MARKET TRENDS & RISK
# ============================================================
with tab2:
    st.header("Community Risk & Growth Profile")
    st.write("Historical price volatility and growth by community (2020–2026).")

    selected_view_community = st.selectbox("Select a community to inspect", sorted(risk_trend_summary['community'].unique()))

    row = risk_trend_summary[risk_trend_summary['community'] == selected_view_community].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Growth (2020–2026)", f"{row['total_growth_pct']:.1f}%")
    col2.metric("Avg Monthly Growth", f"{row['avg_monthly_growth_pct']*100:.2f}%")
    col3.metric("Volatility (Risk)", f"{row['volatility_pct']*100:.2f}%")

    st.subheader("Highest Growth Communities")
    top_growth = risk_trend_summary.sort_values('total_growth_pct', ascending=False).head(10)
    fig, ax = plt.subplots()
    ax.barh(top_growth['community'], top_growth['total_growth_pct'])
    ax.set_xlabel("Total Growth 2020–2026 (%)")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("Highest Volatility (Risk) Communities")
    top_risk = risk_trend_summary.sort_values('volatility_pct', ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    ax2.barh(top_risk['community'], top_risk['volatility_pct']*100, color='indianred')
    ax2.set_xlabel("Volatility (%)")
    ax2.invert_yaxis()
    st.pyplot(fig2)

# ============================================================
# TAB 3: PORTFOLIO OPTIMIZATION
# ============================================================
with tab3:
    st.header("Portfolio Optimization")
    st.write(
        "Given expected return and risk for a set of communities, this finds the "
        "weighting that maximizes risk-adjusted return — using the real correlation "
        "structure between communities' price movements, not just individual volatility."
    )

    st.subheader("Optimized Allocation")
    fig3, ax3 = plt.subplots()
    ax3.pie(portfolio_result['optimized_weight_v2_pct'], labels=portfolio_result['community'], autopct='%1.1f%%')
    st.pyplot(fig3)

    st.subheader("Optimized vs. Equal-Weighted Portfolio")
    st.write("**Equal-weighted:** return=0.860%, risk=1.817%")
    st.write("**Optimized:** return=0.887%, risk=1.727%")
    st.success("The optimized portfolio achieves both higher return and lower risk by exploiting low/negative correlations between communities.")

    st.dataframe(portfolio_result)