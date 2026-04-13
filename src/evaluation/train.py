import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing.preprocess import load_and_preprocess_data

from sklearn.model_selection import GridSearchCV

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
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
        }
    }
    
    results = {}
    
    # Ensure a directory exists for saving models
    os.makedirs('models', exist_ok=True)
    
    for name, config in models.items():
        print(f"Training and Tuning {name}...")
        
        # Create a full pipeline that includes the preprocessor and the model
        base_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
