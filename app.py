"""House Price Prediction System - Streamlit web app.
Loads the pipeline trained in the Colab notebook (house_price_model.pkl) and predicts a price."""
import json
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


model = load_model()
info = load_info()
SALE_YEAR = 2014  # the training data contains sales from 2014

st.title("🏠 House Price Prediction System")
st.caption(
    f"Model: **{info['best_algorithm']}** · trained on {info['n_houses']:,} house sales "
    f"(King County / Seattle area, WA, 2014) · selected with 10-fold cross-validation"
)

st.subheader("Enter the house details")

col1, col2 = st.columns(2)
with col1:
    zipcode = st.selectbox("ZIP code", info["zipcodes"], index=info["zipcodes"].index("98052"))
    bedrooms = st.number_input("Bedrooms", 1, 9, 3)
    bathrooms = st.number_input("Bathrooms", 0.5, 8.0, 2.0, step=0.25)
    floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=0)
    condition = st.slider("Condition (1 = poor, 5 = excellent)", 1, 5, 3)
    view = st.slider("View quality (0 = none, 4 = excellent)", 0, 4, 0)
with col2:
    sqft_living = st.number_input("Living area (sq ft)", 300, 14000, 2000, step=50)
    sqft_basement = st.number_input("Basement area (sq ft)", 0, 5000, 0, step=50)
    sqft_lot = st.number_input("Lot size (sq ft)", 500, 1100000, 7500, step=100)
    year_built = st.number_input("Year built", 1900, 2014, 1985)
    renovated = st.checkbox("Has been renovated")
    waterfront = st.checkbox("Waterfront property")

sqft_above = max(sqft_living - sqft_basement, 0)  # living area = above ground + basement

if st.button("Predict price", type="primary"):
    house = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": sqft_lot, "floors": floors, "waterfront": int(waterfront), "view": view,
        "condition": condition, "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "house_age": SALE_YEAR - year_built, "was_renovated": int(renovated), "zipcode": zipcode,
    }])
    price = max(float(model.predict(house)[0]), 0.0)
    mae = info["cv_mean_mae"]

    st.success(f"### Estimated price: ${price:,.0f}")
    st.write(
        f"Typical error of this model (cross-validated MAE): about **±${mae:,.0f}**, "
        f"so a realistic range is **${max(price - mae, 0):,.0f} – ${price + mae:,.0f}**."
    )
    with st.expander("Show the feature vector sent to the model"):
        st.dataframe(house)

st.divider()
st.caption(
    "Educational project (COMSATS University Islamabad, Lahore – AIC354 Assignment 3). "
    f"Cross-validation RMSE: ${info['cv_mean_rmse']:,.0f}. Prices reflect the 2014 Seattle-area market "
    "and should not be used for real valuations."
)
