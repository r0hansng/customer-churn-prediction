# 📉 Customer Churn Prediction & Retention Strategy

> An AI-driven web application designed to identify at-risk customers and evaluate churn probabilities using classical Machine Learning pipelines. Built for Milestone 1 of **Project 5**.

---

## 📌 Project Overview

This system applies machine learning techniques to historical customer behavior data to predict churn risk and identify key drivers of disengagement.

- **Milestone 1:** Focuses on predictive analytics using Logistic Regression, Decision Trees, and Multi-Layer Perceptrons (MLPs). Features an interactive UI for real-time and batch predictions.
- **Milestone 2 (Upcoming):** Extends the predictive model into an agentic AI retention strategist using LangGraph, capable of reasoning about churn and proposing retention strategies via RAG.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/makeprodigy/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# OR
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
Deploy the interactive web interface locally:
```bash
streamlit run app.py
```
> The application will open automatically at `http://localhost:8501`.

---

## 🏗️ Project Architecture

```text
customer-churn-prediction/
│
├── app.py                   ← Streamlit Dashboard (Single & Batch Predictions)
├── requirements.txt         ← Project dependencies
├── README.md                ← Project documentation
│
├── customer_churn_datasest.csv ← Raw Dataset
├── sample_test.csv          ← Sample file for batch prediction testing
│
├── src/                     
│   ├── preprocess.py        ← Data cleaning & encoding (OneHotEncoding, StandardScaler)
│   └── train.py             ← Model training & evaluation (Accuracy, F1, etc.)
│
└── models/                  ← Pre-trained Scikit-Learn pipelines (.joblib)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    └── mlp.joblib
```

---

## 📊 Core Features

- **Data Preprocessing Pipeline:** Automated handling of missing values, encoding of categorical variables, and scaling of numeric features.
- **Model Evaluation:** Compare performance metrics across Logistic Regression (Best Performing), Decision Trees, and MLPs directly within the app.
- **Single Customer Prediction:** Input a specific customer's profile via sidebar widgets to instantly calculate their churn risk.
- **Batch CSV Prediction:** Upload a dataset to generate bulk predictions. Includes a visual dashboard with churn proportion charts, probability distributions, and a filterable/downloadable results table.

---

## ⚙️ Tech Stack

- **UI Framework:** Streamlit
- **Machine Learning:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Serialization:** Joblib

---

## 📝 Evaluation Criteria (Milestone 1)
This project is designed to meet the Mid-Semester requirements:
- Application of classical ML techniques.
- Effective feature engineering and data preprocessing.
- High usability and interactivity of the UI.
- Comprehensive evaluation metrics reporting.
