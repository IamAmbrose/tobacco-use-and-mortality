# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

# Load models & utils
regressor = joblib.load('regressor.pkl')
classifier = joblib.load('classifier.pkl')
scaler = joblib.load('scaler.pkl')
feature_order = joblib.load('feature_order.pkl')

st.set_page_config(page_title="🚬 Tobacco Mortality ML App", layout="wide")
st.title("🚬 Tobacco Use & Mortality Predictor")

mode = st.sidebar.selectbox("Choose Prediction Mode:", ["Regression (Death Rate)", "Classification (High Fatality)"])

st.sidebar.markdown("---")
st.sidebar.header("Input Features")

# User inputs
smoking_prev = st.sidebar.slider('Smoking Prevalence (%)', 0.0, 50.0, 23.0)
tobacco_price = st.sidebar.number_input('Tobacco Price Index', 0.0, 2000.0, 654.6)
retail_price = st.sidebar.number_input('Retail Prices Index', 0.0, 2000.0, 279.3)
income = st.sidebar.number_input("Real Households' Disposable Income", 0.0, 50000.0, 188.7)
interaction = st.sidebar.number_input('SmokingPrice Interaction', 0.0, 100000.0, 15055.8)
sex_male = st.sidebar.radio('Sex: Male?', ['Yes', 'No']) == 'Yes'
policy_pre2010 = st.sidebar.radio('Policy Era: Pre-2010?', ['Yes', 'No']) == 'Yes'
diag = st.sidebar.text_input('ICD10 Diagnosis', 'All cancers')
diag_type = st.sidebar.text_input('Diagnosis Type', 'All cancers')

# Create input DataFrame
X_input = pd.DataFrame({
    'Smoking Prevalence': [smoking_prev],
    'Tobacco Price Index': [tobacco_price],
    'Retail Prices Index': [retail_price],
    "Real Households' Disposable Income": [income],
    'SmokingPrice_Interaction': [interaction],
    'Sex_Male': [int(sex_male)],
    'Policy_Era_Pre-2010': [int(policy_pre2010)],
    'ICD10 Diagnosis': [diag],
    'Diagnosis Type': [diag_type]
})

# One-hot encode
X_input = pd.get_dummies(X_input, columns=['ICD10 Diagnosis', 'Diagnosis Type'])

# Add any missing columns & reorder
for col in feature_order:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_order]

# Scale
X_scaled = scaler.transform(X_input)

# Mode switch
if mode.startswith("Regression"):
    prediction = regressor.predict(X_scaled)
    st.subheader(f"📈 Predicted Death Rate: **{prediction[0]:.4f}**")
    explainer = shap.Explainer(regressor, X_scaled)
    shap_values = explainer(X_scaled)
    st.subheader("🔍 SHAP Feature Impact (Regression)")
    shap.plots.bar(shap_values[0])
    st.pyplot(bbox_inches='tight')

else:
    prediction = classifier.predict(X_scaled)
    prob = classifier.predict_proba(X_scaled)[0][1]
    result = "⚠️ High Fatality" if prediction[0] == 1 else "✅ Low Fatality"
    st.subheader(f"🔒 Classification: {result}")
    st.write(f"Probability of High Fatality: **{prob:.2%}**")
    explainer = shap.Explainer(classifier, X_scaled)
    shap_values = explainer(X_scaled)
    st.subheader("🔍 SHAP Feature Impact (Classification)")
    shap.plots.bar(shap_values[0])
    st.pyplot(bbox_inches='tight')

st.write("---")
st.caption("Demo: Tobacco Use & Mortality Predictor — powered by Streamlit + RandomForest")
