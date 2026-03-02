# Customer Churn Prediction & Retention Strategy

An ML-driven web application designed to identify at-risk customers and evaluate churn probabilities using classical Machine Learning pipelines. Built as part of Milestone 1 for Project 5.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-predictiongit-mid-sem-milestone-1.streamlit.app/)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/ml-scikit--learn-orange.svg)

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Model Architecture & Optimization](#model-architecture--optimization)
- [Quick Start](#quick-start)
- [Technology Stack](#technology-stack)
- [Evaluation Criteria](#evaluation-criteria)

---

## Overview

This system applies machine learning techniques to historical customer behavior data to predict churn risk and identify key drivers of disengagement. The project is structured into two main milestones:

- **Milestone 1:** Focuses on predictive analytics using Logistic Regression, Decision Trees, and Multi-Layer Perceptrons (MLPs). It features an interactive UI for both real-time and batch predictions.
- **Milestone 2 (Upcoming):** Extends the predictive model into an agentic AI retention strategist utilizing LangGraph. This component will be capable of reasoning about churn and proposing targeted retention strategies via Retrieval-Augmented Generation (RAG).

**Live Environment:** [Customer Churn Prediction App](https://customer-churn-predictiongit-mid-sem-milestone-1.streamlit.app/)

---

## System Architecture

The application is structured thoughtfully to separate concerns among the user interface, data preprocessing, model training, and inference.

```text
customer-churn-prediction/
├── app.py                     # Streamlit entry point (Single & Batch Predictions)
├── requirements.txt           # Project dependencies
├── README.md                  # System documentation
├── customer_churn_datasest.csv# Raw training/validation dataset
├── sample_test.csv            # Sample file for batch prediction validation
├── src/                     
│   ├── preprocess.py          # Data cleaning, normalization, and encoding logic
│   └── train.py               # Model training, validation, and evaluation pipeline
└── models/                    # Serialized Scikit-Learn pipelines (.joblib)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    └── mlp.joblib
```

---

## Core Features

The system offers three distinct capabilities tailored for analytical depth and operational usability:

### 1. Automated Preprocessing Pipeline
Engineered a robust, leak-proof data pipeline utilizing Scikit-Learn's `ColumnTransformer`. It combines `StandardScaler` for continuous numerical features and `OneHotEncoder` for categorical inputs, ensuring data is strictly standardized prior to modeling.

### 2. Real-Time Inference Engine
An interactive prediction dashboard allowing users to manually construct a customer profile across 19 different metrics. The system performs a forward pass and instantly returns a calculated churn probability alongside actionable feature-level insights.

### 3. Batch Evaluation Interface
A bulk processing feature designed for high-throughput evaluation. Users can upload bulk customer profile datasets (CSV) to run the entire dataset through the inference pipeline. The system subsequently generates a dynamic dashboard featuring interactive distributions, probability histograms, and exportable result artifacts.

---

## Model Architecture & Optimization

To ensure optimal predictive performance and prevent overfitting, the underlying models were tuned using `GridSearchCV` incorporating 3-fold cross-validation:

- **Logistic Regression:** Optimized primarily over inverse regularization strengths (`C`) and optimization algorithms (`solver`).
- **Decision Trees:** Constrained structurally by tuning `max_depth` and `min_samples_leaf` to ensure generalizability.
- **Multi-Layer Perceptrons (MLPs):** Tuned across varying hidden layer topologies and L2 weight decay (`alpha`) parameters.

---

## Quick Start

### Prerequisites
- Python 3.8 or higher.
- `pip` package manager.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/makeprodigy/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Initialize a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # macOS / Linux
   # .venv\Scripts\activate        # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   *The application will bind to `localhost:8501` by default.*

---

## Technology Stack

- **Application Framework:** Streamlit
- **Machine Learning Library:** Scikit-Learn
- **Data Engineering:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib

---

## Evaluation Criteria (Milestone 1)

This system was designed to address the mid-semester requirements:
- Implementation and validation of classical ML algorithms.
- Rigorous feature engineering and disciplined data preprocessing.
- High usability, accessibility, and interactivity within the UI layer.
- Comprehensive and transparent evaluation metrics reporting.