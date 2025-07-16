import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ✅ CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="🚬 Tobacco & Mortality", layout="wide")
st.title("🚬 Tobacco Use & Mortality — DASHBOARD")

# ----------------------------------------------------------
# ✅ LOAD MODELS & DATA
# ----------------------------------------------------------
pipeline_rate = joblib.load("pipeline_regressor_rate.pkl")
pipeline_fat = joblib.load("pipeline_regressor_fatality.pkl")

df = pd.read_csv("merged_dataset.csv")
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
df = df.rename(columns=lambda x: x.replace(' ', '_'))
df["Sex"] = df["Sex"].astype(str).str.strip().str.title()

df["SmokingPrice_Interaction"] = df["Smoking_Prevalence"] * df["Tobacco_Price_Index"]
df["Policy_Era_Pre-2010"] = (df["Year"] < 2010).astype(int)
df["Sex_Male"] = df["Sex"].apply(lambda x: 1 if x == 'Male' else 0)

if "Death_Rate" not in df.columns:
    df["Death_Rate"] = df["Value_fat"] / df["Value_adm"]

diagnosis_options = sorted(df["ICD10_Diagnosis"].dropna().unique())
diagnosis_type_options = sorted(df["Diagnosis_Type"].dropna().unique())

# ----------------------------------------------------------
# ✅ TABS
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 EDA", "🧮 Predict"])

# ----------------------------------------------------------
# ✅ OVERVIEW
# ----------------------------------------------------------
with tab1:
    st.header("📌 Project Overview")
    st.markdown("""
    This dashboard predicts **Death Rate** or **Raw Fatalities**.
    ✅ Debug mode: shows **full input + raw prediction**.
    """)
    st.metric("Latest Smoking Prevalence (%)", df["Smoking_Prevalence"].iloc[-1])
    st.metric("Latest Tobacco Price Index", df["Tobacco_Price_Index"].iloc[-1])
    st.metric("Latest Fatalities", df["Value_fat"].dropna().iloc[-1])

# ----------------------------------------------------------
# ✅ EDA
# ----------------------------------------------------------
with tab2:
    st.header("📊 Exploratory Data Analysis")

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x="Year", y="Smoking_Prevalence", hue="Sex", marker="o", ax=ax1)
    ax1.set_title("Smoking Prevalence by Sex")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x="Year", y="Value_adm", hue="Sex", marker="o", ax=ax2)
    ax2.set_title("Admissions by Sex")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df, x="Year", y="Value_fat", hue="Sex", marker="o", ax=ax3)
    ax3.set_title("Fatalities by Sex")
    st.pyplot(fig3)

# ----------------------------------------------------------
# ✅ PREDICT (DEBUG)
# ----------------------------------------------------------
with tab3:
    st.header("🧮 Predict")

    st.sidebar.header("Prediction Mode")
    mode = st.sidebar.radio("Choose Prediction:", ["Death Rate", "Raw Fatalities"])

    st.sidebar.header("Input Features")
    smoking_prev = st.sidebar.slider("Smoking Prevalence (%)", 0.0, 50.0, 23.0)
    tobacco_price = st.sidebar.number_input("Tobacco Price Index", 0.0, 2000.0, 654.6)
    retail_price = st.sidebar.number_input("Retail Prices Index", 0.0, 2000.0, 279.3)
    income = st.sidebar.number_input("Real Households' Disposable Income", 0.0, 50000.0, 188.7)
    sex_male = st.sidebar.radio("Sex: Male?", ["Yes", "No"]) == "Yes"
    policy_pre2010 = st.sidebar.radio("Policy Era: Pre-2010?", ["Yes", "No"]) == "Yes"
    diag = st.sidebar.selectbox("ICD10 Diagnosis", diagnosis_options)
    diag_type = st.sidebar.selectbox("Diagnosis Type", diagnosis_type_options)

    # ✅ Fix casing to match training data
    diag = diag.strip()
    diag_type = diag_type.strip()

    interaction = smoking_prev * tobacco_price

    if mode == "Raw Fatalities":
        value_adm = st.sidebar.number_input("Admissions Count (Value_Adm)", 0, 5000000, 50000)
    else:
        value_adm = None

    # ✅ Input DataFrame
    input_data = {
        "Smoking_Prevalence": [smoking_prev],
        "Tobacco_Price_Index": [tobacco_price],
        "Retail_Prices_Index": [retail_price],
        "Real_Households'_Disposable_Income": [income],
        "SmokingPrice_Interaction": [interaction],
        "Sex_Male": [int(sex_male)],
        "Policy_Era_Pre-2010": [int(policy_pre2010)],
        "ICD10_Diagnosis": [diag],
        "Diagnosis_Type": [diag_type]
    }

    if mode == "Raw Fatalities":
        input_data["Value_adm"] = [value_adm]

    X_input = pd.DataFrame(input_data)

    st.info(f"🗂️ **Mode:** {mode}")
    st.write("✅ Final input to pipeline:")
    st.write(X_input)

    # ✅ Predict
    if mode == "Death Rate":
        prediction = pipeline_rate.predict(X_input)[0]
        mean_val = df["Death_Rate"].mean()
    else:
        prediction = pipeline_fat.predict(X_input)[0]
        mean_val = df["Value_fat"].mean()

    st.success(f"✅ Raw prediction: {prediction:.4f}")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Predicted", "Historical Mean"], [prediction, mean_val], color=["blue", "gray"])
    ax.set_title(f"{mode}: Predicted vs Historical Mean")
    ax.set_ylabel(mode)
    st.pyplot(fig)

st.write("---")
st.caption("Built by **IamAmbrose** — Streamlit, sklearn, seaborn, matplotlib 🚀")
