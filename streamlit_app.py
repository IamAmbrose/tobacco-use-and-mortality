import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

#  CONFIG

st.set_page_config(page_title="🚬 Tobacco Use & Mortality Dashboard", layout="wide")
st.title("🚬 Tobacco Use & Mortality — FINAL PREDICTIVE DASHBOARD")

#  LOAD MODELS & DATA
regressor = joblib.load("regressor.pkl")
feature_order = joblib.load("feature_order.pkl")
df = pd.read_csv("merged_dataset.csv")

#  Clean column names (just strip, no title-casing!)
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
df.rename(columns={"Value_fat": "Value_Fat", "Value_adm": "Value_Adm"}, inplace=True)
df["Sex"] = df["Sex"].astype(str).str.strip().str.title()

#  Extract dropdown options directly from feature_order
icd10_cols = [c for c in feature_order if c.startswith("ICD10 Diagnosis_")]
diag_type_cols = [c for c in feature_order if c.startswith("Diagnosis Type_")]

diagnosis_options = sorted(list(set(c.replace("ICD10 Diagnosis_", "") for c in icd10_cols)))
diagnosis_type_options = sorted(list(set(c.replace("Diagnosis Type_", "") for c in diag_type_cols)))

#  TABS
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 EDA", "📈 Predict"])

# OVERVIEW

with tab1:
    st.header("📌 Project Overview")
    st.markdown(
        """
        This dashboard analyzes **tobacco use**, **pricing**, **household income**, and **mortality trends** in the UK (2004–2015).

        👉 Predict **Death Rate** robustly  
        👉 Compare trends for **Male vs Female**
        """
    )
    st.metric("Latest Smoking Prevalence (%)", df["Smoking Prevalence"].iloc[-1])
    st.metric("Latest Tobacco Price Index", df["Tobacco Price Index"].iloc[-1])
    st.metric("Latest Fatalities", df["Value_Fat"].dropna().iloc[-1])

#  EDA
with tab2:
    st.header("📊 Exploratory Data Analysis")

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x="Year", y="Smoking Prevalence", hue="Sex", marker="o", ax=ax1)
    ax1.set_title("Smoking Prevalence Over Time by Sex")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x="Year", y="Value_Adm", hue="Sex", marker="o", ax=ax2)
    ax2.set_title("Admissions Over Time by Sex")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x="Year", y="Value_Fat", hue="Sex", marker="o", ax=ax3)
    ax3.set_title("Fatalities Over Time by Sex")
    st.pyplot(fig3)

    presc_trend = (
        df.groupby(["Year", "Sex"])["All Pharmacotherapy Prescriptions"]
        .mean().reset_index()
    )

    fig4, ax4 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=presc_trend, x="Year", y="All Pharmacotherapy Prescriptions",
                 hue="Sex", marker="o", ax=ax4)
    ax4.set_title("Average Prescriptions Over Time by Sex")
    st.pyplot(fig4)

#  PREDICT TAB 
with tab3:
    st.header("📈 Predict Death Rate")

    st.sidebar.header("Input Features")

    smoking_prev = st.sidebar.slider("Smoking Prevalence (%)", 0.0, 50.0, 23.0)
    tobacco_price = st.sidebar.number_input("Tobacco Price Index", 0.0, 2000.0, 654.6)
    retail_price = st.sidebar.number_input("Retail Prices Index", 0.0, 2000.0, 279.3)
    income = st.sidebar.number_input("Real Households' Disposable Income", 0.0, 50000.0, 188.7)
    interaction = st.sidebar.number_input("SmokingPrice Interaction", 0.0, 100000.0, 15055.8)
    sex_male = st.sidebar.radio("Sex: Male?", ["Yes", "No"]) == "Yes"
    policy_pre2010 = st.sidebar.radio("Policy Era: Pre-2010?", ["Yes", "No"]) == "Yes"

    diag = st.sidebar.selectbox("ICD10 Diagnosis", diagnosis_options)
    diag_type = st.sidebar.selectbox("Diagnosis Type", diagnosis_type_options)

    X_input = pd.DataFrame({
        "Smoking Prevalence": [smoking_prev],
        "Tobacco Price Index": [tobacco_price],
        "Retail Prices Index": [retail_price],
        "Real Households' Disposable Income": [income],
        "SmokingPrice_Interaction": [interaction],
        "Sex_Male": [int(sex_male)],
        "Policy_Era_Pre-2010": [int(policy_pre2010)],
        "ICD10 Diagnosis": [diag],
        "Diagnosis Type": [diag_type]
    })

    X_input = pd.get_dummies(X_input, columns=["ICD10 Diagnosis", "Diagnosis Type"])

    # Fill missing dummy columns
    for col in feature_order:
        if col not in X_input.columns:
            X_input[col] = 0

    X_input = X_input[feature_order].astype(float)

    st.write("✅ Input preview:", X_input)
    st.write("✅ Sum of row (should include dummy):", X_input.sum(axis=1))

    prediction = regressor.predict(X_input)[0]
    st.subheader(f"📈 Predicted Death Rate: **{prediction:.2f}**")

    mean_fatalities = df["Value_Fat"].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Predicted", "Historical Mean"], [prediction, mean_fatalities], color=["blue", "gray"])
    ax.set_ylabel("Death Rate")
    ax.set_title("Predicted vs Historical Mean Death Rate")
    st.pyplot(fig)

st.write("---")
st.caption("Built by **Ambrose** with Streamlit, Matplotlib, Seaborn & Random Forests.")
