# Evaluation Module

This module handles model training, evaluation, and hyperparameter tuning.

## Contents:
- `train.py` - Main training and evaluation functions
  - `train_and_evaluate()` - Trains models and performs grid search tuning

## Usage:
```python
from src.evaluation.train import train_and_evaluate

results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
```
