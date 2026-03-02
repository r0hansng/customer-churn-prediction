# 📉 Customer Churn Prediction & Retention Strategy

> An AI-driven web application designed to identify at-risk customers and evaluate churn probabilities using classical Machine Learning pipelines. Built for Milestone 1 of **Project 5**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-predictiongit-mid-sem-milestone-1.streamlit.app/)

---

## 📌 Project Overview

This system applies machine learning techniques to historical customer behavior data to predict churn risk and identify key drivers of disengagement.

- **Milestone 1:** Focuses on predictive analytics using Logistic Regression, Decision Trees, and MLPs. Features an interactive UI for real-time and batch predictions.
- **Milestone 2 (Upcoming):** Extends the predictive model into an agentic AI retention strategist using LangGraph, capable of reasoning about churn and proposing retention strategies via RAG.

---

## 🚀 Live Demo
**Access the fully deployed application here:**  
🔗 [**Customer Churn Prediction App**](https://customer-churn-predictiongit-mid-sem-milestone-1.streamlit.app/)

---

## 🎬 Project Demo Video
**Watch the full demonstration here:**  
🔗 [**Demo Video (Google Drive)**](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

---

## 📄 Technical Report
**Full LaTeX report (Overleaf, read-only):**  
🔗 [**Technical Report — Milestone 1**](https://www.overleaf.com/read/kdrjyhtsnxhs#4b6dcb)

---

## 💻 Local Quick Start

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

## 🏗️ System Architecture

![Customer Churn Prediction — System Architecture](GenAI_architecture_diagram.png)

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

## 📊 Core Features (3 Distinct Sub-Features)

1. **Automated Scikit-Learn Preprocessing Pipeline:** Engineered a robust, leak-proof data pipeline combining `StandardScaler` for continuous numerical features and `OneHotEncoder` for categoricals, managed cleanly via a `ColumnTransformer`.
2. **Real-time Single Customer Inference Engine:** An interactive prediction dashboard where agents can manually toggle 19 different customer profile metrics and instantly receive a calculated churn probability score alongside actionable business insights.
3. **Batch Prediction Analytics Dashboard:** A bulk processing feature allowing users to upload CSV datasets. The system runs entire datasets through the inference engine and generates a dynamic dashboard featuring interactive bar charts, probability distrbution histograms, and a filterable/downloadable results table.

---

## ⚙️ Model Optimization
To ensure the highest possible performance and to satisfy the project's optimization criteria, models were fine-tuned using `GridSearchCV` with 3-fold cross-validation. 
- **Logistic Regression** was optimized over varying inverse regularization strengths (`C`) and solvers.
- **Decision Trees** were constrained by optimizing `max_depth` and `min_samples_leaf` to prevent overfitting.
- **MLPs** were tuned across hidden layer sizes and L2 penalty (`alpha`) parameters.

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
