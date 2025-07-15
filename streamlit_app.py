# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Config
st.set_page_config(page_title="🚬 Tobacco Mortality Dashboard", layout="wide")
st.title("🚬 Tobacco Use & Mortality — Interactive Dashboard")

# Load trained models & scaler
regressor = joblib.load('regressor.pkl')
classifier = joblib.load('classifier.pkl')
scaler = joblib.load('scaler.pkl')
feature_order = joblib.load('feature_order.pkl')

# Load your dataset for EDA
df = pd.read_csv("merged_featured.csv")
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

# Tabs — just like Olympics app
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 EDA", "🤖 Predict"])

#  Tab 1 — Overview
with tab1:
    st.header("📌 Project Overview")
    st.markdown("""
    This project analyzes the relationship between **tobacco use**, **pricing**, **household income**, and **mortality** in the UK from **2004–2015**.
    
    **Key features:**
    - Smoking Prevalence
    - Tobacco & Retail Price Indices
    - Real Disposable Income
    - ICD10 Diagnosis, Fatalities & Admissions
    
    **Purpose:**  
    👉 Forecast mortality rates (`Death_Rate`) using regression.  
    👉 Classify diagnoses as **High vs. Low Fatality**.  
    👉 Visualize how policy & price changes impact outcomes.
    
    🔍 Explore EDA or switch to **Predict** to test scenarios!
    """)

    # Add headline stats
    st.metric("Latest Smoking Prevalence (%)", df['Smoking Prevalence'].iloc[-1])
    st.metric("Latest Tobacco Price Index", df['Tobacco Price Index'].iloc[-1])
    st.metric("Latest Death Rate", df['Death_Rate'].dropna().iloc[-1])

#  Tab 2 — EDA
with tab2:
    st.header("📊 Exploratory Data Analysis")

    # Trendline: Smoking Prevalence over time
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=df, x='Year', y='Smoking Prevalence', marker='o', ax=ax1)
    ax1.set_title("Smoking Prevalence Over Time")
    st.pyplot(fig1)

    # Correlation heatmap for numeric cols
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

    # Distribution of Death Rate
    fig3, ax3 = plt.subplots()
    sns.histplot(df['Death_Rate'].dropna(), bins=30, kde=True, ax=ax3, color="red")
    ax3.set_title("Death Rate Distribution")
    st.pyplot(fig3)

#  Tab 3 — Predict
with tab3:
    st.header("🤖 Predict Mortality / Fatality")

    mode = st.sidebar.selectbox(
        "Choose Prediction Mode:",
        ["Regression (Death Rate)", "Classification (High Fatality)"]
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Input Features")

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

    X_input = pd.get_dummies(X_input, columns=['ICD10 Diagnosis', 'Diagnosis Type'])
    for col in feature_order:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[feature_order]

    X_scaled = scaler.transform(X_input)

    if mode.startswith("Regression"):
        prediction = regressor.predict(X_scaled)
        st.subheader(f"📈 Predicted Death Rate: **{prediction[0]:.4f}**")

        explainer = shap.Explainer(regressor, X_input)
        shap_values = explainer(X_scaled)

        st.subheader("🔍 SHAP Feature Impact (Regression)")
        fig4, ax4 = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig4)

    else:
        prediction = classifier.predict(X_scaled)
        prob = classifier.predict_proba(X_scaled)[0][1]
        result = "⚠️ High Fatality" if prediction[0] == 1 else "✅ Low Fatality"

        st.subheader(f"🔒 Classification: {result}")
        st.write(f"Probability of High Fatality: **{prob:.2%}**")

        explainer = shap.Explainer(classifier, X_input)
        shap_values = explainer(X_scaled)

        st.subheader("🔍 SHAP Feature Impact (Classification)")
        fig5, ax5 = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig5)

st.write("---")
st.caption("Built by Ambrose using Streamlit, SHAP, Seaborn & Matplotlib")
