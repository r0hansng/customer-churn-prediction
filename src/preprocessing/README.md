# Preprocessing Module

This module handles data loading, cleaning, and preprocessing operations.

## Contents:
- `preprocess.py` - Main preprocessing functions
  - `load_and_preprocess_data()` - Loads data and applies preprocessing transformations

## Usage:
```python
from src.preprocessing.preprocess import load_and_preprocess_data

X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('data/customer_churn_datasest.csv')
```
