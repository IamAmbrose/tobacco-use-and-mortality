import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#  CONFIG
st.set_page_config(page_title="🚬 Tobacco Use & Mortality Dashboard", layout="wide")
st.title("🚬 Tobacco Use & Mortality — Interactive Dashboard")

# LOAD MODELS

regressor = joblib.load('regressor.pkl')
classifier = joblib.load('classifier.pkl')
feature_order = joblib.load('feature_order.pkl')

#  LOAD DATA
df = pd.read_csv("merged_dataset.csv")

# Clean up column names: remove newlines, extra spaces, title case
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip().str.replace('_', '_').str.title()

df.rename(columns={
    'Value_fat': 'Value_fat',
    'Value_adm': 'Value_adm'
}, inplace=True)

df['Sex'] = df['Sex'].astype(str).str.strip().str.title()

#  TABS
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 EDA", "🤖 Predict"])

#  Overview
with tab1:
    st.header("📌 Project Overview")
    st.markdown("""
    This dashboard analyzes **tobacco use**, **pricing**, **household income**, and **mortality trends** in the UK (2004–2015).

    **Features:** Smoking Prevalence, Tobacco/Retail Prices, Income, ICD10 Diagnosis.

    👉 Predict **Death Rate** or classify **High vs Low Fatality**  
    👉 Compare trends for **Male vs Female**
    """)

    st.metric("Latest Smoking Prevalence (%)", df['Smoking Prevalence'].iloc[-1])
    st.metric("Latest Tobacco Price Index", df['Tobacco Price Index'].iloc[-1])
    st.metric("Latest Fatalities", df['Value_Fat'].dropna().iloc[-1])

#  EDA

with tab2:
    st.header("📊 Exploratory Data Analysis")

    st.write("✅ Unique Sex values:", df['Sex'].unique())

    # ✅ Smoking Prevalence
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x='Year', y='Smoking Prevalence', hue='Sex', marker='o', ax=ax1)
    ax1.set_title("Smoking Prevalence Over Time by Sex")
    st.pyplot(fig1)

    # ✅ Admissions
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x='Year', y='Value_Adm', hue='Sex', marker='o', ax=ax2)
    ax2.set_title("Admissions Over Time by Sex")
    st.pyplot(fig2)

    # ✅ Fatalities
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x='Year', y='Value_Fat', hue='Sex', marker='o', ax=ax3)
    ax3.set_title("Fatalities Over Time by Sex")
    st.pyplot(fig3)

    # ✅ Prescriptions
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df, x='Year', y='All Pharmacotherapy Prescriptions', hue='Sex', marker='o', ax=ax4)
    ax4.set_title("Prescriptions Over Time by Sex")
    st.pyplot(fig4)


#  Predict
with tab3:
    st.header("🤖 Predict Mortality / Fatality")

    mode = st.sidebar.selectbox(
        "Choose Prediction Mode:",
        ["Regression (Death Rate)", "Classification (High Fatality)"]
    )

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
        'Smokingprice Interaction': [interaction],
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
    X_input = X_input.astype(float)

    if mode.startswith("Regression"):
        prediction = regressor.predict(X_input)[0]
        st.subheader(f"📈 Predicted Death Rate: **{prediction:.2f}**")

        mean_fatalities = df['Value_fat'].mean()
        fig, ax = plt.subplots()
        ax.bar(['Predicted', 'Historical Mean'], [prediction, mean_fatalities], color=['blue', 'gray'])
        ax.set_ylabel("Fatalities")
        ax.set_title("Predicted vs Historical Mean Fatalities")
        st.pyplot(fig)

    else:
        prediction = classifier.predict(X_input)[0]
        prob = classifier.predict_proba(X_input)[0][1]

        if prediction == 1:
            result_text = "**🔒 Prediction: HIGH FATALITY DEATHS**"
            icon = "⚠️"
        else:
            result_text = "**✅ Prediction: LOW FATALITY DEATHS**"
            icon = "✅"

        st.subheader(f"{icon} {result_text}")
        st.write(f"**Probability of High Fatality Deaths:** `{prob:.2%}`")

        fig, ax = plt.subplots()
        ax.bar(['High Fatality Probability'], [prob], color='red')
        ax.axhline(0.5, color='gray', linestyle='--', label='Threshold')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("High Fatality Probability")
        ax.legend()
        st.pyplot(fig)


st.write("---")
st.caption("Built by **Ambrose** with Streamlit, Matplotlib, Seaborn & Random Forests.")
