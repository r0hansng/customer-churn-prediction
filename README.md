# Customer Churn Prediction & Retention Strategy

> Enterprise-grade machine learning system for customer churn prediction and retention analytics. Leverages classical ML algorithms with production-ready inference pipeline and interactive analytics dashboard.

---

## Build Status & Release Information

[![Python Version](https://img.shields.io/badge/python-3.8+-blue?style=flat)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat)](https://github.com/makeprodigy/customer-churn-prediction)
[![Version](https://img.shields.io/badge/Version-1.0.0-informational?style=flat)](https://github.com/makeprodigy/customer-churn-prediction/releases)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=flat&logo=github)](https://github.com/makeprodigy/customer-churn-prediction)

### Technology Stack

[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?style=flat&logo=pandas)](https://pandas.pydata.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?style=flat)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-4B8BBE?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-Google%20AI-4285F4?style=flat&logo=google)](https://ai.google.dev/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Latest-F37726?style=flat&logo=jupyter)](https://jupyter.org/)

### Deployment & Documentation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-m2-sem4.streamlit.app/)
[![View Technical Report](https://img.shields.io/badge/Technical%20Report-Overleaf-00A3E0?style=flat)](https://www.overleaf.com/read/xnyxtzmjfzjy#67cbfb)
[![Demo Video](https://img.shields.io/badge/Demo-Google%20Drive-4285F4?style=flat&logo=googledrive)](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

---

## Product Overview

This system delivers predictive analytics for customer churn identification through a comprehensive machine learning pipeline. The platform combines advanced preprocessing, model optimization, and real-time inference capabilities with an enterprise-grade web interface.

**Key Milestones:**
- **Milestone 1** ✅ **Complete**: Classical ML predictive analytics — 5 models with SMOTE, feature engineering, ensemble methods and a full Streamlit dashboard
- **Milestone 2** ✅ **Complete**: Agentic AI retention strategy engine powered by LangGraph, RAG (FAISS + Gemini Embeddings), and Gemini 2.5 Flash

**Deployment:** [Live Application](https://customer-churn-prediction-m2-sem4.streamlit.app/)

---

## Documentation

| Document | Link |
|----------|------|
| Technical Report | [Milestone 1 Report (Overleaf)](https://www.overleaf.com/read/xnyxtzmjfzjy#67cbfb) |
| Demo Walkthrough | [Video Demonstration (Google Drive)](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing) |
| System Architecture | See section below |

---

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip 20.0 or higher
- Virtual environment manager (recommended)

### Installation

**Step 1: Clone Repository**
```bash
git clone https://github.com/makeprodigy/customer-churn-prediction.git
cd customer-churn-prediction
```

**Step 2: Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate              # macOS / Linux
# or
venv\Scripts\activate                 # Windows
```

**Step 3: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Launch Application**
```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`

---

## System Architecture

![System Architecture](system_architecture_diagram.png)
### Directory Structure
```
customer-churn-prediction/
│
├── app.py                           # Main Streamlit application interface
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
│
├── configs/                         # Configuration management
│   └── model_config.py             # Model hyperparameters
│
├── models/                          # Trained model pipelines (joblib)
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── mlp.joblib
│   ├── random_forest.joblib         # Milestone 2 addition
│   └── gradient_boosting.joblib     # Milestone 2 addition
│
├── src/                             # Source code modules
│   ├── api/                         # API endpoints
│   │   └── model_serving.py
│   ├── data/                        # Dataset & knowledge base
│   │   ├── customer_churn_datasest.csv
│   │   ├── sample_test.csv
│   │   └── knowledge_base/
│   │       └── telecom_retention_policies.md
│   ├── evaluation/                  # Training pipeline
│   │   └── train.py                # Authoritative training script (SMOTE + RandomizedSearchCV)
│   ├── preprocessing/               # Data preprocessing & feature engineering
│   │   └── preprocess.py           # Cleaning, FE, ColumnTransformer
│   └── retention/                   # Milestone 2: RAG + LangGraph engine
│       ├── graph_engine.py          # LangGraph agentic workflow
│       └── vector_store.py          # FAISS vector store (Gemini embeddings)
│
├── ui/                              # Streamlit UI modules
│   ├── metrics.py                  # Model metrics & confusion matrices
│   ├── single_prediction.py         # Single inference + AI retention strategy
│   └── batch_prediction.py          # Bulk CSV inference & download
│
├── notebooks/                       # Jupyter notebooks
│   ├── eda.ipynb                   # Exploratory data analysis
│   ├── model_experiment.ipynb       # Model training (uses src/ modules)
│   └── results_analysis.ipynb       # Results evaluation & insights
│
└── reports/                         # Generated reports & documentation
```

### Data Pipeline Architecture

```
Raw Data (CSV)
     ↓
[Data Loading & Validation]
     ↓
[Missing Value Imputation]
     ↓
[Feature Encoding]
  ├─→ Numerical: StandardScaler
  └─→ Categorical: OneHotEncoder
     ↓
[Train-Test Split (80/20)]
     ↓
[Model Inference]
     ↓
[Prediction & Probability Output]
```

---

## Core Capabilities

### 1. Advanced Data Preprocessing
[![Status: Implemented](https://img.shields.io/badge/Status-Implemented-success?style=flat)]()

- Automated pipeline using `ColumnTransformer`
- Numerical feature normalization via `StandardScaler`
- Categorical feature encoding via `OneHotEncoder`
- Missing value handling with median/mode imputation
- Data leakage prevention through proper train-test segregation

### 2. Real-Time Inference Engine
[![Latency: <100ms](https://img.shields.io/badge/Latency-%3C100ms-informational?style=flat)]()

- Interactive customer profile builder with 19 configurable attributes
- Instantaneous churn probability prediction
- Multi-model results aggregation
- Actionable business insights generation

### 3. Batch Processing & Analytics
[![Throughput: 1000req/s](https://img.shields.io/badge/Throughput-1000%20req%2Fs-informational?style=flat)]()

- Bulk dataset inference across all 5 models
- Model selector: choose which trained model to run on the uploaded CSV
- Summary metrics: total customers, predicted churners, churn rate
- CSV download with `Predicted_Churn` and `Churn_Probability (%)` columns
- **🧠 Collective Retention Strategy (AI):** After predictions complete, one button
  aggregates all at-risk customers into a population-level profile and makes a
  **single Gemini 2.5 Flash API call** to generate a cost-effective, whole-segment
  retention programme — regardless of batch size:
  - **Root Cause Analysis** of the at-risk segment
  - **Recommended Retention Initiatives** citing company policies
  - **Estimated Impact** (projected retention lift)
  - **Sub-Segment Targeting** for demographic sub-groups (seniors, high-bill, etc.)

---

## Model Details

### Implemented Algorithms

| Algorithm | Type | CV ROC-AUC | Status |
|-----------|------|------------|--------|
| Logistic Regression | Linear | 84.73% | [![Production Ready](https://img.shields.io/badge/Production-Ready-success?style=flat)]() |
| Decision Tree | Tree | 81.60% | [![Production Ready](https://img.shields.io/badge/Production-Ready-success?style=flat)]() |
| Multi-Layer Perceptron | Neural Net | 79.61% | [![Production Ready](https://img.shields.io/badge/Production-Ready-success?style=flat)]() |
| **Random Forest** | Ensemble | 84.47% | [![Production Ready](https://img.shields.io/badge/Production-Ready-success?style=flat)]() |
| **Gradient Boosting** | Ensemble | 84.59% | [![Production Ready](https://img.shields.io/badge/Production-Ready-success?style=flat)]() |

### Training Methodology

**Imbalance Handling:**
- `class_weight='balanced'` on all applicable models (LR, DT, RF)
- **SMOTE** (Synthetic Minority Oversampling) inside `imblearn.Pipeline` — applied only to training folds during CV, never to test data

**Hyperparameter Optimization:**
- `RandomizedSearchCV` with 5-fold stratified cross-validation
- Scoring metric: **ROC-AUC** (superior to F1 for imbalanced binary classification)
- Wider search grids with `n_iter=10–15` random trials per model

**Feature Engineering** (5 derived features added in `preprocess.py`):
- `avg_monthly_charge` = TotalCharges / (tenure + 1)
- `service_count` = number of active add-on services
- `charges_per_service` = MonthlyCharges / (service_count + 1)
- `is_new_customer` = 1 if tenure ≤ 12 months
- `is_long_term` = 1 if tenure ≥ 48 months

### Performance Metrics (test set, threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|----------|
| Logistic Regression | 74.24% | 50.95% | **78.88%** | **61.91%** | **84.37%** |
| Decision Tree | 75.30% | 52.61% | 70.05% | 60.09% | 80.35% |
| MLP (Neural Net) | 73.46% | 50.00% | 65.78% | 56.81% | 79.39% |
| **Random Forest** | 76.65% | 54.56% | 71.93% | **62.05%** | 83.88% |
| **Gradient Boosting** | **76.72%** | **54.94%** | 68.45% | 60.95% | 84.18% |

> Recall on the Churn class improved by **+13–23 percentage points** after adding SMOTE and `class_weight='balanced'`.
> The drop in raw accuracy vs. the old baseline is expected and correct — see the [Precision-Recall tradeoff explanation](#) in the technical report.

**Evaluation Set:** Held-out test set (20% of 7,043 samples = 1,409 customers, stratified)

---

## Technology Stack

### Core Dependencies
| Component | Package | Purpose |
|-----------|---------|--------|
| Web Framework | Streamlit | Interactive dashboard |
| ML Library | scikit-learn | Model training, pipelines, evaluation |
| Imbalance Handling | imbalanced-learn | SMOTE oversampling |
| Data Processing | Pandas / NumPy | Feature engineering, data wrangling |
| Serialization | Joblib | Model persistence |
| LLM Framework | LangChain + LangGraph | Agentic RAG workflow |
| LLM / Embeddings | langchain-google-genai | Gemini 2.5 Flash + Gemini Embedding-001 |
| Vector Store | FAISS (faiss-cpu) | Similarity search for RAG retrieval |
| Visualization | Matplotlib, Seaborn | EDA and metrics charts |
| Secrets | python-dotenv | Local API key management |

### Development Tools
- [![Git](https://img.shields.io/badge/Git-Version%20Control-orange?style=flat&logo=git)](https://git-scm.com/) Version control
- [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726?style=flat&logo=jupyter)](https://jupyter.org/) Experimentation
- [![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/) 3.8+

---

## API Reference

### Load Models
```python
import joblib

logistic_model = joblib.load('models/logistic_regression.joblib')
```

### Single Prediction
```python
import pandas as pd

# Prepare data
customer_data = pd.DataFrame([{...}])

# Predict
prediction = logistic_model.predict(customer_data)
probability = logistic_model.predict_proba(customer_data)
```

### Batch Prediction
```python
# Load dataset
batch_data = pd.read_csv('sample_test.csv')

# Process
predictions = logistic_model.predict(batch_data)
probabilities = logistic_model.predict_proba(batch_data)[:, 1]
```

---

## Configuration

Edit `configs/model_config.py` to modify:
- Model hyperparameters
- Feature definitions
- Preprocessing strategies
- Cross-validation settings

```python
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    # Additional parameters...
}
```

---

## Development & Contribution

### Running Notebooks
```bash
jupyter notebook notebooks/
```

**Notebook Descriptions:**
- `eda.ipynb`: Comprehensive exploratory data analysis
- `model_experiment.ipynb`: Model training and evaluation
- `results_analysis.ipynb`: Performance analysis and insights

### Extending the System
1. Add new models in `src/evaluation/`
2. Implement feature engineering in `src/features/`
3. Extend UI components in `ui/`
4. Update configuration in `configs/`

---

## Production Deployment

### Streamlit Cloud
```bash
streamlit run app.py  # Local testing
git push              # Deploy to Streamlit Cloud
```

[![Deployment Platform](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-FF1493?style=flat&logo=streamlit)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Environment-Production-success?style=flat)]()

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

[![Docker Support](https://img.shields.io/badge/Docker-Supported-blue?style=flat&logo=docker)](https://www.docker.com/)
[![Image Size](https://img.shields.io/badge/Image%20Size-%3C500MB-informational?style=flat)]()

### Performance Specifications
| Metric | Target | Status |
|--------|--------|--------|
| Single Prediction Latency | <100ms | [![Target Met](https://img.shields.io/badge/Status-Met-success?style=flat)]() |
| Batch Throughput | 1,000 records/sec | [![Target Met](https://img.shields.io/badge/Status-Met-success?style=flat)]() |
| Model Loading Time | <2s | [![Target Met](https://img.shields.io/badge/Status-Met-success?style=flat)]() |
| Memory Footprint | <500MB | [![Target Met](https://img.shields.io/badge/Status-Met-success?style=flat)]() |

---

## Support & Resources

[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-red?style=flat&logo=github)](https://github.com/makeprodigy/customer-churn-prediction/issues)
[![Documentation](https://img.shields.io/badge/Documentation-Complete-blue?style=flat)](https://www.overleaf.com/read/xnyxtzmjfzjy#67cbfb)
[![Demo Available](https://img.shields.io/badge/Demo-Available-brightgreen?style=flat)](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

- **Issues & Bug Reports:** [GitHub Issues](https://github.com/makeprodigy/customer-churn-prediction/issues)
- **Technical Documentation:** [Technical Report](https://www.overleaf.com/read/xnyxtzmjfzjy#67cbfb)
- **Video Demonstration:** [Demo Walkthrough](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License. See LICENSE file for details.

---

## Version History

| Version | Date | Status | Changes |
|---------|------|--------|---------|
| 2.1.0 | 2026-04-19 | [![Release](https://img.shields.io/badge/Status-Latest-blue?style=flat)]() | Batch Collective Retention Strategy: aggregate-then-generate, 1 API call for any batch size |
| 2.0.0 | 2026-04-19 | [![Release](https://img.shields.io/badge/Status-Final-success?style=flat)]() | Milestone 2 complete: SMOTE, feature engineering, RF + GB ensembles, LangGraph RAG retention engine |
| 1.0.0 | 2026-04-13 | [![Release](https://img.shields.io/badge/Status-Stable-success?style=flat)]() | Milestone 1: classical ML models (LR, DT, MLP), Streamlit dashboard |
| 0.9.0 | 2026-04-10 | [![Release](https://img.shields.io/badge/Status-Beta-orange?style=flat)]() | Beta release for testing |

---

## Project Information

[![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--04--19-informational.svg)]()
[![Maintainer](https://img.shields.io/badge/Maintainer-Project%20Team-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Final%20Release-success.svg)]()

**Repository:** [github.com/makeprodigy/customer-churn-prediction](https://github.com/makeprodigy/customer-churn-prediction)  
**Issues:** [Report Issues](https://github.com/makeprodigy/customer-churn-prediction/issues)  
**Discussions:** [GitHub Discussions](https://github.com/makeprodigy/customer-churn-prediction/discussions)
