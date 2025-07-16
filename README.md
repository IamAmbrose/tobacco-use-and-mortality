# 🚬 Tobacco Use and Mortality (2004–2015) — ML & Data Analysis Project

[![Streamlit](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://tobacco-use-and-mortality-sjw2jdt3fk5xy5tuypkn9e.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/IamAmbrose/tobacco-use-and-mortality)

---

This project analyzes the relationship between **tobacco use**, **pricing**, **income**, and **mortality** in the UK (2004–2015).  
It combines **data cleaning**, **EDA**, **feature engineering**, **machine learning**, and an **interactive Streamlit dashboard** for real-time scenario testing.

---

## 🎯 **Project Objectives**

- Analyze trends: smoking prevalence, pricing, income, admissions, fatalities.
- Predict **Death Rate** (*fatalities per admissions*).
- Predict **Raw Fatalities** (*total estimated deaths*).
- Deploy a **live Streamlit app** with clear inputs & visuals.

---

## 🗂️ **Data Sources**

- Annual UK surveys & administrative health data:
  - Smoking prevalence
  - Tobacco Price Index, Retail Prices Index
  - Real household disposable income
  - ICD10 diagnoses & types
  - Admissions (`Value_adm`)
  - Fatalities (`Value_fat`)
  - Derived features: `SmokingPrice_Interaction`

---

## 🔍 **Key Processing**

1️⃣ **Cleaning**
- Renamed columns for consistency (handled `Value_fat` variations).
- Standardized casing, stripped spaces, fixed line breaks.

2️⃣ **Feature Engineering**
- `SmokingPrice_Interaction` to capture price & usage effect.
- Binary feature: `Policy_Era_Pre-2010`.

3️⃣ **Model Training**
- **Death Rate**
  - Pipeline RMSE: ~0.03
  - R²: ~0.97
- **Raw Fatalities**
  - Pipeline RMSE: ~1784.58
  - R²: ~0.99

4️⃣ **Deployment**
- Streamlit app with tabs:
  - 📌 Overview
  - 📊 EDA
  - 📈 Predict (Death Rate or Raw Fatalities)

---

## 📊 **Key Insights**

- Higher tobacco prices reduce smoking prevalence.
- Lower smoking prevalence reduces admissions and mortality.
- Mortality risks vary by **diagnosis type** & **policy era**.
- Gender differences are visible in trends.
- Clear interaction between pricing and household income.

---

## 🧩 **Streamlit Dashboard**

**Features:**
- Medium-sized visuals for Smoking Prevalence, Admissions, Fatalities, Prescriptions.
- Interactive input sliders, radio buttons, dropdowns for:
  - Smoking prevalence, pricing, income.
  - ICD10 Diagnosis & Type.
  - Sex & Policy Era.
- **Prediction Modes:**
  - **Death Rate**: expected deaths per admission.
  - **Raw Fatalities**: total expected number of deaths.
- Visual bar chart comparing prediction vs historical mean.

---

## ⚙️ **How to Run**

1️⃣ Install:
```bash
pip install -r requirements.txt


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
