# 🛒 Favorita Grocery Sales Forecasting

## 📌 Project Overview

This project focuses on **time series forecasting** for the Favorita Grocery dataset. It includes comprehensive **data cleaning, model development, interpretability with SHAP, and an interactive deployment using Streamlit**.

The goal is to accurately predict future grocery sales by leveraging various statistical and machine learning models, and deliver insights in a user-friendly, real-time app.

---

## ✅ Key Features

- 🧹 **Data Cleaning & Preprocessing**
  - Handled missing values using **KNN imputation**, **mean**, **median**, and **mode** strategies
  - Created lag features and rolling statistics for temporal context

- 📈 **Model Development**
  - Trained multiple models:
    - **SARIMAX** (with exogenous variables)
    - **Random Forest Regressor**
    - **XGBoost Regressor**
  - Applied **GridSearchCV** for hyperparameter tuning
  - Built full **ML pipelines** for model reproducibility

- 🧠 **Model Interpretability**
  - Used **SHAP (SHapley Additive exPlanations)** to explain model predictions and feature importance

- 🚀 **Deployment**
  - Built and deployed an interactive **Streamlit web application**
  - Integrated **user input prediction**, real-time results, and **dynamic visualizations**
  - Visual components include:
    - Bar, box, scatter plots
    - Selectbox & slider filters
    - SQL-style playground (if applicable)

---

## 📊 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python (Pandas, NumPy)** | Data wrangling |
| **scikit-learn** | ML model pipeline, tuning |
| **XGBoost** | High-performing regression |
| **SARIMAX (statsmodels)** | Time series modeling |
| **SHAP** | Model explainability |
| **Streamlit** | Web app deployment |
| **Matplotlib, Seaborn, Plotly** | Data visualization |

---

## 🚀 How to Run

1. Clone the repository  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt
