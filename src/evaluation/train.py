import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.preprocessing.preprocess import load_and_preprocess_data

MODEL_CONFIGS = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_leaf': [1, 5, 10]
        }
    },
    'MLP': {
        'model': MLPClassifier(max_iter=1000, random_state=42),
        'params': {
            'classifier__hidden_layer_sizes': [(50,), (100,)],
            'classifier__alpha': [0.0001, 0.001]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'classifier__n_estimators':     [100, 200, 300],
            'classifier__max_depth':        [5, 10, None],
            'classifier__min_samples_leaf': [1, 5, 10],
            'classifier__class_weight':     ['balanced', None],
        }
    },
    'Gradient Boosting': {
        # GradientBoostingClassifier does not support class_weight natively.
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators':  [100, 200, 300],
            'classifier__learning_rate': [0.03, 0.05, 0.1, 0.2],
            'classifier__max_depth':     [3, 5, 7],
            'classifier__subsample':     [0.7, 0.8, 1.0],
        }
    },
}


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    results = {}

    # Ensure a directory exists for saving models
    os.makedirs('models', exist_ok=True)

    for name, config in MODEL_CONFIGS.items():
        print(f"Training and Tuning {name}...")

        # Create a full pipeline that includes the preprocessor and the model
        base_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])

        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=config['params'],
            scoring='f1',
            cv=3,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_

        y_pred = best.predict(X_test)
        results[name] = {
            'model':     best,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }

        model_filename = f"models/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best, model_filename)
        print(f"  Saved → {model_filename}")

    return results
