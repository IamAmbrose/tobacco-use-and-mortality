import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="✅ FINAL TEST", layout="centered")
st.title("✅ FINAL HARD CODED TEST")

# Load model + feature order
regressor = joblib.load("regressor.pkl")
feature_order = joblib.load("feature_order.pkl")

# Hardcoded input — SAME as your working inspect script
X_input = pd.DataFrame({
    "Smoking Prevalence": [23],
    "Tobacco Price Index": [654.6],
    "Retail Prices Index": [279.3],
    "Real Households' Disposable Income": [188.7],
    "SmokingPrice_Interaction": [15055.8],
    "Sex_Male": [1],
    "Policy_Era_Pre-2010": [1],
    "ICD10 Diagnosis": ["All cancers"],
    "Diagnosis Type": ["All cancers"]
})

# Encode dummies
X_input = pd.get_dummies(X_input, columns=["ICD10 Diagnosis", "Diagnosis Type"])

# Fill missing
for col in feature_order:
    if col not in X_input.columns:
        X_input[col] = 0

X_input = X_input[feature_order].astype(float)

# Debug
st.write("✅ Active dummies:", [c for c in X_input.columns if X_input[c].iloc[0] == 1])
st.write("✅ Row sum:", X_input.sum(axis=1))

# Predict
prediction = regressor.predict(X_input)[0]
st.subheader(f"✅ FINAL TEST PREDICTION: {prediction:.6f}")


st.write("---")
st.caption("✅ Built by **Ambrose** — Streamlit, Matplotlib, Seaborn, Random Forest ")

