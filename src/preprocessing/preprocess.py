import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Service columns whose "Yes" value indicates an active add-on
_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven derived features that expose patterns the raw columns
    cannot express on their own.

    New numeric features
    --------------------
    avg_monthly_charge   : TotalCharges / (tenure + 1)
                           — lifetime average spend; high-tenure / low-spend
                             customers differ from new high-spend ones.
    service_count        : number of active (Yes) add-on services
                           — customers with many services are harder to replace.
    charges_per_service  : MonthlyCharges / (service_count + 1)
                           — how much a customer pays per active service;
                             high values signal potential over-billing risk.
    is_new_customer      : 1 if tenure <= 12 months, else 0
                           — new customers churn at disproportionately high rates.
    is_long_term         : 1 if tenure >= 48 months, else 0
                           — long-tenure customers are significantly more loyal.
    """
    X = X.copy()

    # 1. Average monthly spend over customer lifetime
    X["avg_monthly_charge"] = X["TotalCharges"] / (X["tenure"] + 1)

    # 2. Count of active add-on services
    X["service_count"] = X[_SERVICE_COLS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # 3. Monthly spend per active service (proxy for value-per-service)
    X["charges_per_service"] = X["MonthlyCharges"] / (X["service_count"] + 1)

    # 4. Customer lifecycle flags
    X["is_new_customer"] = (X["tenure"] <= 12).astype(int)
    X["is_long_term"]    = (X["tenure"] >= 48).astype(int)

    return X


def load_and_preprocess_data(filepath='customer_churn_datasest.csv'):
    # Load dataset
    df = pd.read_csv(filepath)

    # Drop customerID as it's not a useful feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # TotalCharges is object, need to convert to numeric, coercing errors to NaN
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Feature engineering
    X = _engineer_features(X)

    # Identify numerical and categorical columns
    numeric_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'avg_monthly_charge', 'service_count', 'charges_per_service',
        'is_new_customer', 'is_long_term',
    ]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor
