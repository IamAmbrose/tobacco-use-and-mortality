# 🚬 Tobacco Use and Mortality (2004–2015) — ML & Data Analysis Project

[![Streamlit](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://tobacco-use-and-mortality-sjw2jdt3fk5xy5tuypkn9e.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/IamAmbrose/tobacco-use-and-mortality)

---


This project analyzes the relationship between tobacco use, economic factors, and mortality rates in the UK from **2004–2015**.  
It combines **data cleaning, exploratory analysis, feature engineering, machine learning**, and a **live Streamlit dashboard** to predict and explain mortality patterns related to smoking.

---

## 📈 **Project Objectives**

- Analyze trends in tobacco use, price indices, income, and mortality.
- Predict **mortality rate (`Death_Rate`)** using regression.
- Classify diagnoses into **High vs Low Fatality** based on historical fatality counts.
- Deploy an **interactive web app** for real-time scenario testing.

---

## 🗂️ **Data Sources**

- Annual survey and administrative health data covering:
  - Smoking prevalence
  - Tobacco Price Index & Retail Prices Index
  - Real household disposable income
  - ICD10 diagnoses
  - Admissions & fatalities (`Value_adm`, `Value_fat`)
  - Engineered interaction features

---

## 🔬 **Processes**

### ✅ 1. Data Cleaning & Preparation
- Removed missing target rows for both regression and classification tasks.
- Cleaned column names to fix hidden line breaks (`\n`, `\r`).
- Encoded categorical variables (`ICD10 Diagnosis`, `Diagnosis Type`).
- Standardized numeric features for ML modeling.

### ✅ 2. Feature Engineering
- Created `SmokingPrice_Interaction` to capture the combined impact of smoking prevalence and price.
- Binarized `Value_fat` into `High_Fatality` class using the median.

### ✅ 3. Exploratory Analysis
- Confirmed trends: increasing tobacco prices, declining smoking rates.
- Noted high fatality risk in cancers & circulatory diseases.
- Detected clear correlations between price, prevalence, and mortality.

### ✅ 4. Model Training
- **Regression:**
  - Linear Regression RMSE: **3967.65**, R²: **0.9971**
  - Random Forest RMSE: **2466.61**, R²: **0.9989**
- **Classification:**
  - Random Forest Accuracy: **98%**
  - ROC AUC: **0.9993**
  - Confusion Matrix:
    ```
    [[67  0]
     [ 3 80]]
    ```
  - Precision, Recall, F1 all at **0.98+**

### ✅ 5. Deployment
- Packaged as a **Streamlit app** with:
  - Mode switch: Regression or Classification
  - User inputs for all key factors
  - Real-time predictions with **SHAP** explanations for feature impact

---

## 🏆 **Key Insights**

- Mortality rates strongly depend on:
  - Smoking prevalence & intensity
  - Tobacco pricing
  - Household income
  - Policy era & demographics
  - Specific diagnoses

- Effective pricing & policy can reduce smoking prevalence and lower smoking-related deaths.

---

## 📌 **Project Performance**

| Model | RMSE | R² | Accuracy | ROC AUC |
|-------|------|-----|----------|---------|
| Linear Regression | 3967.65 | 0.9971 | — | — |
| Random Forest Regression | 2466.61 | 0.9989 | — | — |
| Random Forest Classification | — | — | 98% | 0.9993 |

---

## ⚙️ **How to Run**

1️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
````

2️⃣ **(Optional) Retrain models:**

```bash
python train_models.py
```

3️⃣ **Launch the Streamlit app:**

```bash
streamlit run app.py
```

---

## 🎉 **Conclusion**

This project shows how **data + ML + explainability** can help policymakers, researchers, and health professionals understand and reduce tobacco-related mortality.
The tool supports data-driven decisions about smoking policy, pricing, and public health strategy.

---

**Author:** [IamAmbrose](https://github.com/IamAmbrose)

---

📂 **Repo:** [https://github.com/IamAmbrose/tobacco-use-and-mortality](https://github.com/IamAmbrose/tobacco-use-and-mortality)

````

---

## ✅ **How to use it**

1️⃣ Copy the full text above.  
2️⃣ Save it as `README.md` in your GitHub repo root folder.  
3️⃣ Push your changes:
```bash
git add README.md
git commit -m "Add project README with summary and results"
git push origin main
````

---
