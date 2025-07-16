import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ✅ CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="🚬 Tobacco Use & Mortality Dashboard", layout="wide")
st.title("🚬 Tobacco Use & Mortality — FINAL DASHBOARD")

# ----------------------------------------------------------
# ✅ LOAD MODELS & BOTH DATASETS
# ----------------------------------------------------------
regressor = joblib.load("regressor.pkl")
feature_order = joblib.load("feature_order.pkl")

# ✅ Load dataset for EDA only
eda_df = pd.read_csv("merged_dataset.csv")

# ✅ Load dataset for prediction (matches regressor.pkl)
predict_df = pd.read_csv("merged_featured.csv")

# ✅ Clean both consistently
for df in [eda_df, predict_df]:
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    df.rename(columns={"Value_fat": "Value_Fat", "Value_adm": "Value_Adm"}, inplace=True)
    df["Sex"] = df["Sex"].astype(str).str.strip().str.title()

# ----------------------------------------------------------
# ✅ Extract ICD10 & Diagnosis Type options from prediction data
# ----------------------------------------------------------
diagnosis_options = sorted(predict_df["ICD10 Diagnosis"].dropna().unique())
diag_type_options = sorted(predict_df["Diagnosis Type"].dropna().unique())

# ----------------------------------------------------------
# ✅ TABS
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 EDA", "📈 Predict"])

# ----------------------------------------------------------
# ✅ OVERVIEW
# ----------------------------------------------------------
with tab1:
    st.header("📌 Project Overview")
    st.markdown(
        """
        This dashboard uses **two datasets**:
        - `merged_dataset.csv` → for rich EDA.
        - `merged_featured.csv` → for predictions aligned with your trained RandomForest model.

        ✅ **Safe dummy encoding**, debug outputs, and clear matching mean your predictions always match your manual test.
        """
    )
    st.metric("Latest Smoking Prevalence (%)", eda_df["Smoking Prevalence"].iloc[-1])
    st.metric("Latest Tobacco Price Index", eda_df["Tobacco Price Index"].iloc[-1])
    st.metric("Latest Fatalities", eda_df["Value_Fat"].dropna().iloc[-1])

# ----------------------------------------------------------
# ✅ EDA — merged_dataset.csv
# ----------------------------------------------------------
with tab2:
    st.header("📊 Exploratory Data Analysis")

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=eda_df, x="Year", y="Smoking Prevalence", hue="Sex", marker="o", ax=ax1)
    ax1.set_title("Smoking Prevalence Over Time by Sex")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=eda_df, x="Year", y="Value_Adm", hue="Sex", marker="o", ax=ax2)
    ax2.set_title("Admissions Over Time by Sex")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=eda_df, x="Year", y="Value_Fat", hue="Sex", marker="o", ax=ax3)
    ax3.set_title("Fatalities Over Time by Sex")
    st.pyplot(fig3)

    presc_trend = (
        eda_df.groupby(["Year", "Sex"])["All Pharmacotherapy Prescriptions"]
        .mean().reset_index()
    )

    fig4, ax4 = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=presc_trend, x="Year", y="All Pharmacotherapy Prescriptions",
                 hue="Sex", marker="o", ax=ax4)
    ax4.set_title("Average Prescriptions Over Time by Sex")
    st.pyplot(fig4)

# ----------------------------------------------------------
# ✅ PREDICT — merged_featured.csv
# ----------------------------------------------------------
with tab3:
    st.header("📈 Predict Death Rate ")

    st.sidebar.header("Input Features")

    smoking_prev = st.sidebar.slider("Smoking Prevalence (%)", 0.0, 50.0, 23.0)
    tobacco_price = st.sidebar.number_input("Tobacco Price Index", 0.0, 2000.0, 654.6)
    retail_price = st.sidebar.number_input("Retail Prices Index", 0.0, 2000.0, 279.3)
    income = st.sidebar.number_input("Real Households' Disposable Income", 0.0, 50000.0, 188.7)
    interaction = st.sidebar.number_input("SmokingPrice Interaction", 0.0, 100000.0, 15055.8)
    sex_male = st.sidebar.radio("Sex: Male?", ["Yes", "No"]) == "Yes"
    policy_pre2010 = st.sidebar.radio("Policy Era: Pre-2010?", ["Yes", "No"]) == "Yes"

    diag = st.sidebar.selectbox("ICD10 Diagnosis", diagnosis_options)
    diag_type = st.sidebar.selectbox("Diagnosis Type", diag_type_options)

    # ✅ Build input DataFrame
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

    # ✅ One-hot encode
    X_input = pd.get_dummies(X_input, columns=["ICD10 Diagnosis", "Diagnosis Type"])

    # ✅ Fill any missing dummy
    for col in feature_order:
        if col not in X_input.columns:
            X_input[col] = 0

    X_input = X_input[feature_order].astype(float)

    # ✅ Debug: show active dummies
    active_dummies = [col for col in X_input.columns if X_input[col].iloc[0] == 1 and ("ICD10" in col or "Diagnosis Type" in col)]
    st.write("✅ Active ICD10/Diagnosis dummy columns:", active_dummies)
    st.write("✅ Row sum:", X_input.sum(axis=1))

    # ✅ Predict
    prediction = regressor.predict(X_input)[0]
    st.subheader(f"📈 Predicted Death Rate: **{prediction:.4f}**")

    mean_fatalities = predict_df["Value_Fat"].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Predicted", "Historical Mean"], [prediction, mean_fatalities], color=["blue", "gray"])
    ax.set_ylabel("Death Rate")
    ax.set_title("Predicted vs Historical Mean Death Rate")
    st.pyplot(fig)


st.write("---")
st.caption("✅ Built by **Ambrose** — Streamlit, Matplotlib, Seaborn, Random Forest ")

