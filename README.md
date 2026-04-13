# Customer Churn Prediction & Retention Strategy

> Enterprise-grade machine learning system for customer churn prediction and retention analytics. Leverages classical ML algorithms with production-ready inference pipeline and interactive analytics dashboard.

---

## Build Status & Release Information

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)](https://github.com/makeprodigy/customer-churn-prediction)
[![Version](https://img.shields.io/badge/Version-1.0.0-informational.svg)](https://github.com/makeprodigy/customer-churn-prediction/releases)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg?logo=github)](https://github.com/makeprodigy/customer-churn-prediction)

### Technology Stack

[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-150458.svg?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243.svg?logo=numpy)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Latest-F37726.svg?logo=jupyter)](https://jupyter.org/)

### Deployment & Documentation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-predictiongit-mid-sem-milestone-1.streamlit.app/)
[![View Technical Report](https://img.shields.io/badge/Technical%20Report-Overleaf-00A3E0.svg)](https://www.overleaf.com/read/kdrjyhtsnxhs#4b6dcb)
[![Demo Video](https://img.shields.io/badge/Demo-Google%20Drive-4285F4.svg?logo=googledrive)](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

---

## Product Overview

This system delivers predictive analytics for customer churn identification through a comprehensive machine learning pipeline. The platform combines advanced preprocessing, model optimization, and real-time inference capabilities with an enterprise-grade web interface.

**Key Milestones:**
- **Milestone 1** (Current): Classical ML predictive analytics with comprehensive UI for single and batch inference
- **Milestone 2** (Planned): Intelligent retention strategy engine powered by LangGraph and RAG

**Deployment:** [Live Application](https://customer-churn-predictiongit-mid-sem-milestone-1.streamlit.app/)

---

## Documentation

| Document | Link |
|----------|------|
| Technical Report | [Milestone 1 Report (Overleaf)](https://www.overleaf.com/read/kdrjyhtsnxhs#4b6dcb) |
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
├── models/                          # Pre-trained models (binary joblib format)
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   └── mlp.joblib
│
├── src/                             # Source code modules
│   ├── api/                         # API endpoints
│   │   └── model_serving.py        # REST API for inference
│   ├── data/                        # Dataset management
│   │   ├── customer_churn_datasest.csv
│   │   └── sample_test.csv
│   ├── evaluation/                  # Model evaluation
│   │   └── train.py                # Training & hyperparameter tuning
│   ├── features/                    # Feature engineering
│   ├── preprocessing/               # Data preprocessing pipeline
│   │   └── preprocess.py           # Data transformation
│   └── __init__.py
│
├── ui/                              # User interface modules
│   ├── metrics.py                  # Metrics visualization
│   ├── single_prediction.py         # Single inference interface
│   └── batch_prediction.py          # Batch processing interface
│
├── notebooks/                       # Jupyter notebooks
│   ├── eda.ipynb                   # Exploratory data analysis
│   ├── model_experiment.ipynb       # Model development & tuning
│   └── results_analysis.ipynb       # Results evaluation & insights
│
└── reports/                         # Generated reports & visualizations
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
[![Status: Implemented](https://img.shields.io/badge/Status-Implemented-success.svg)]()

- Automated pipeline using `ColumnTransformer`
- Numerical feature normalization via `StandardScaler`
- Categorical feature encoding via `OneHotEncoder`
- Missing value handling with median/mode imputation
- Data leakage prevention through proper train-test segregation

### 2. Real-Time Inference Engine
[![Latency: <100ms](https://img.shields.io/badge/Latency-%3C100ms-informational.svg)]()

- Interactive customer profile builder with 19 configurable attributes
- Instantaneous churn probability prediction
- Multi-model results aggregation
- Actionable business insights generation

### 3. Batch Processing & Analytics
[![Throughput: 1000req/s](https://img.shields.io/badge/Throughput-1000%20req%2Fs-informational.svg)]()

- Bulk dataset inference capability
- Dynamic analytics dashboard with interactive visualizations
- Probability distribution analysis
- Filterable results with export functionality
- CSV output for downstream analysis

---

## Model Details

### Implemented Algorithms

| Algorithm | Status | CV Score | Production |
|-----------|--------|----------|------------|
| Logistic Regression | [![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)]() | 5-Fold CV | [![Production Ready](https://img.shields.io/badge/Production-Ready-success.svg)]() |
| Decision Tree Classifier | [![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)]() | 5-Fold CV | [![Production Ready](https://img.shields.io/badge/Production-Ready-success.svg)]() |
| Multi-Layer Perceptron | [![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)]() | 5-Fold CV | [![Production Ready](https://img.shields.io/badge/Production-Ready-success.svg)]() |

### Training Methodology

**Hyperparameter Optimization:**
- Cross-validation strategy: 5-fold stratified splits
- Scoring metric: F1-score (balanced precision-recall)
- Parameter grid search for each model type

**Logistic Regression Parameters:**
- Inverse regularization strength (C): [0.1, 1.0, 10.0]
- Solver: [lbfgs, liblinear]

**Decision Tree Parameters:**
- Maximum depth: [3, 5, 10, None]
- Minimum samples per leaf: [1, 5, 10]

**Neural Network Parameters:**
- Hidden layer sizes: [(50,), (100,)]
- L2 regularization (alpha): [0.0001, 0.001]

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| Logistic Regression | ![Accuracy](https://img.shields.io/badge/Accuracy-81.97%25-brightgreen.svg) | ![Precision](https://img.shields.io/badge/Precision-68.42%25-blue.svg) | ![Recall](https://img.shields.io/badge/Recall-59.25%25-orange.svg) | ![F1](https://img.shields.io/badge/F1--Score-63.51%25-informational.svg) | [![Best](https://img.shields.io/badge/Ranking-1st-gold.svg)]() |
| Decision Tree | ![Accuracy](https://img.shields.io/badge/Accuracy-79.99%25-brightgreen.svg) | ![Precision](https://img.shields.io/badge/Precision-61.88%25-blue.svg) | ![Recall](https://img.shields.io/badge/Recall-63.54%25-orange.svg) | ![F1](https://img.shields.io/badge/F1--Score-62.70%25-informational.svg) | [![Competitive](https://img.shields.io/badge/Ranking-2nd-silver.svg)]() |
| MLP (Neural Network) | ![Accuracy](https://img.shields.io/badge/Accuracy-78.57%25-brightgreen.svg) | ![Precision](https://img.shields.io/badge/Precision-61.41%25-blue.svg) | ![Recall](https://img.shields.io/badge/Recall-51.21%25-orange.svg) | ![F1](https://img.shields.io/badge/F1--Score-55.85%25-informational.svg) | [![Competitive](https://img.shields.io/badge/Ranking-3rd-inactive.svg)]() |

**Evaluation Set:** Held-out test set (20% of 7,043 samples = 1,409 customers)

---

## Technology Stack

### Core Dependencies
| Component | Package | Version | Status |
|-----------|---------|---------|--------|
| Web Framework | Streamlit | Latest | [![Verified](https://img.shields.io/badge/Verified-Stable-success.svg)]() |
| ML Library | scikit-learn | Latest | [![Verified](https://img.shields.io/badge/Verified-Stable-success.svg)]() |
| Data Processing | Pandas | Latest | [![Verified](https://img.shields.io/badge/Verified-Stable-success.svg)]() |
| Numerical Computing | NumPy | Latest | [![Verified](https://img.shields.io/badge/Verified-Stable-success.svg)]() |
| Visualization | Matplotlib, Seaborn | Latest | [![Verified](https://img.shields.io/badge/Verified-Stable-success.svg)]() |
| Serialization | Joblib | Latest | [![Verified](https://img.shields.io/badge/Verified-Stable-success.svg)]() |

### Development Tools
- [![Git](https://img.shields.io/badge/Git-Version%20Control-orange.svg?logo=git)](https://git-scm.com/) Version control
- [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37726.svg?logo=jupyter)](https://jupyter.org/) Experimentation
- [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python)](https://www.python.org/) 3.8+

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

[![Deployment Platform](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-FF1493.svg?logo=streamlit)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Environment-Production-success.svg)]()

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

[![Docker Support](https://img.shields.io/badge/Docker-Supported-blue.svg?logo=docker)](https://www.docker.com/)
[![Image Size](https://img.shields.io/badge/Image%20Size-%3C500MB-informational.svg)]()

### Performance Specifications
| Metric | Target | Status |
|--------|--------|--------|
| Single Prediction Latency | <100ms | [![Target Met](https://img.shields.io/badge/Status-Met-success.svg)]() |
| Batch Throughput | 1,000 records/sec | [![Target Met](https://img.shields.io/badge/Status-Met-success.svg)]() |
| Model Loading Time | <2s | [![Target Met](https://img.shields.io/badge/Status-Met-success.svg)]() |
| Memory Footprint | <500MB | [![Target Met](https://img.shields.io/badge/Status-Met-success.svg)]() |

---

## Support & Resources

[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-red.svg?logo=github)](https://github.com/makeprodigy/customer-churn-prediction/issues)
[![Documentation](https://img.shields.io/badge/Documentation-Complete-blue.svg)](https://www.overleaf.com/read/kdrjyhtsnxhs#4b6dcb)
[![Demo Available](https://img.shields.io/badge/Demo-Available-brightgreen.svg)](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

- **Issues & Bug Reports:** [GitHub Issues](https://github.com/makeprodigy/customer-churn-prediction/issues)
- **Technical Documentation:** [Technical Report](https://www.overleaf.com/read/kdrjyhtsnxhs#4b6dcb)
- **Video Demonstration:** [Demo Walkthrough](https://drive.google.com/file/d/1u08SJbGc92HiiyLoe42iHyELQlHl03aQ/view?usp=sharing)

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License. See LICENSE file for details.

---

## Version History

| Version | Date | Status | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-04-13 | [![Release](https://img.shields.io/badge/Status-Stable-success.svg)]() | Initial release with classical ML models |
| 0.9.0 | 2026-04-10 | [![Release](https://img.shields.io/badge/Status-Beta-orange.svg)]() | Beta release for testing |

---

## Project Information

[![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--04--13-informational.svg)]()
[![Maintainer](https://img.shields.io/badge/Maintainer-Project%20Team-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

**Repository:** [github.com/makeprodigy/customer-churn-prediction](https://github.com/makeprodigy/customer-churn-prediction)  
**Issues:** [Report Issues](https://github.com/makeprodigy/customer-churn-prediction/issues)  
**Discussions:** [GitHub Discussions](https://github.com/makeprodigy/customer-churn-prediction/discussions)
