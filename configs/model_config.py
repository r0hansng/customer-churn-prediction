# Configuration file for model training and evaluation
# Add your configuration parameters here

MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "models": {
        "logistic_regression": {
            "max_iter": 1000,
        },
        "decision_tree": {
            "random_state": 42,
        },
        "mlp": {
            "max_iter": 1000,
            "random_state": 42,
        }
    }
}

DATA_CONFIG = {
    "numeric_features": ['tenure', 'MonthlyCharges', 'TotalCharges'],
    "target_column": "Churn"
}
