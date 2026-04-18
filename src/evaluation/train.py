"""
Training pipeline for the Customer Churn Prediction project.

Improvements applied
--------------------
1. class_weight='balanced'   — penalises misclassifying the minority Churn class.
2. roc_auc scoring           — appropriate for imbalanced binary classification.
3. SMOTE                     — synthetic minority oversampling inside the CV loop
                               (imblearn Pipeline ensures SMOTE is applied only to
                               training folds, never to validation or test data).
4. RandomizedSearchCV        — efficiently explores wider hyperparameter spaces.
5. Two new model families    — Random Forest and Gradient Boosting ensembles.
6. Threshold tuning report   — metrics at both 0.50 and 0.35 decision thresholds.

Usage (from project root):
    python -m src.evaluation.train
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path resolution — works from any working directory
# ---------------------------------------------------------------------------
_SRC_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.preprocess import load_and_preprocess_data

# ---------------------------------------------------------------------------
# Model configurations
# Each entry: classifier, hyperparameter search grid, number of random trials
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "classifier__C":            [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver":       ["lbfgs", "liblinear"],
            "classifier__class_weight": ["balanced", None],
        },
        "n_iter": 15,
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "classifier__max_depth":        [3, 5, 7, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
            "classifier__class_weight":     ["balanced", None],
        },
        "n_iter": 15,
    },
    "mlp": {
        # MLPClassifier does not support class_weight; SMOTE + roc_auc handle imbalance.
        "model": MLPClassifier(max_iter=1000, random_state=42),
        "params": {
            "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50), (50, 25)],
            "classifier__alpha":              [0.0001, 0.001, 0.01],
            "classifier__learning_rate_init": [0.001, 0.01],
        },
        "n_iter": 10,
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators":     [100, 200, 300],
            "classifier__max_depth":        [5, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10],
            "classifier__class_weight":     ["balanced", None],
        },
        "n_iter": 15,
    },
    "gradient_boosting": {
        # GradientBoostingClassifier does not support class_weight natively.
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_depth":     [3, 5, 7],
            "classifier__subsample":     [0.7, 0.8, 1.0],
        },
        "n_iter": 15,
    },
}

DECISION_THRESHOLD = 0.35   # lower than 0.50 to favour Recall on churn class


# ---------------------------------------------------------------------------
# Threshold-aware metric reporter
# ---------------------------------------------------------------------------
def _metrics_at_threshold(model, X_test, y_test, threshold: float):
    """Return a dict of common metrics evaluated at a given decision threshold."""
    proba  = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred":    y_pred,
    }


# ---------------------------------------------------------------------------
# Main training & evaluation function
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """
    For each model:
      1. Build an imblearn Pipeline: Preprocessor → SMOTE → Classifier
      2. Tune with RandomizedSearchCV (scoring=roc_auc, cv=5)
      3. Evaluate on the held-out test set at both 0.50 and 0.35 thresholds
      4. Save the best pipeline to models/<name>.joblib

    Returns a results dict keyed by model name.
    """
    models_dir = os.path.join(_PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    results = {}

    for name, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Training : {name.replace('_', ' ').title()}")
        print(f"{'='*60}")

        # Build imblearn Pipeline (SMOTE only fires during .fit(), not .predict())
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote",         SMOTE(random_state=42)),
            ("classifier",    config["model"]),
        ])

        # RandomizedSearchCV with roc_auc — better for imbalanced classification
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Evaluate at default (0.50) and tuned (0.35) thresholds
        m50 = _metrics_at_threshold(best, X_test, y_test, threshold=0.50)
        m35 = _metrics_at_threshold(best, X_test, y_test, threshold=DECISION_THRESHOLD)

        results[name] = {
            "model":        best,
            "best_params":  search.best_params_,
            "cv_roc_auc":   search.best_score_,
            # Store threshold=0.50 result as the canonical metric set
            **{k: m50[k] for k in ["accuracy","precision","recall","f1","roc_auc","confusion_matrix"]},
            "metrics_35": m35,
        }

        # ---- Print ----
        print(f"  Best params   : {search.best_params_}")
        print(f"  CV ROC-AUC    : {search.best_score_:.4f}")
        print()
        print(f"  {'Metric':<12}  {'@ 0.50':>8}  {'@ 0.35':>8}")
        print(f"  {'-'*32}")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            v50 = m50[metric]
            v35 = m35[metric]
            print(f"  {metric:<12}  {v50:>7.2%}  {v35:>7.2%}")
        print(f"\n  Confusion matrix (threshold=0.50):\n{m50['confusion_matrix']}")
        print(f"\n  Classification report (threshold=0.50):")
        print(classification_report(y_test, m50["y_pred"],
                                    target_names=["No Churn", "Churn"]))

        # Save the pipeline (predict() will use sklearn's default 0.50 threshold)
        out_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(best, out_path)
        print(f"  Saved → {out_path}")

    # ---- Summary table ----
    print("\n" + "="*60)
    print("  FINAL COMPARISON  (test set, threshold = 0.50)")
    print("="*60)
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name.replace("_", " ").title(),
            "Accuracy":  f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Recall":    f"{r['recall']*100:.2f}%",
            "F1":        f"{r['f1']*100:.2f}%",
            "ROC-AUC":   f"{r['roc_auc']*100:.2f}%",
        })
    print(pd.DataFrame(rows).set_index("Model").to_string())
    print()

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = os.path.join(
        _PROJECT_ROOT, "src", "data", "customer_churn_datasest.csv"
    )
    print(f"Loading data from : {data_path}")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(data_path)
    print(f"Train size : {X_train.shape[0]} samples | {X_train.shape[1]} features")
    print(f"Test  size : {X_test.shape[0]} samples")
    print(f"Churn rate (train) : {y_train.mean()*100:.1f}%")
    print(f"Churn rate (test)  : {y_test.mean()*100:.1f}%")

    train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
    print("✅  All models trained, evaluated, and saved.")
