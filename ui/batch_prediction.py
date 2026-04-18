import streamlit as st
import pandas as pd
import io

try:
    from src.preprocessing.preprocess import _engineer_features
except ImportError:
    _SERVICE_COLS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    def _engineer_features(X):
        X = X.copy()
        X["avg_monthly_charge"]  = X["TotalCharges"] / (X["tenure"] + 1)
        X["service_count"]       = X[_SERVICE_COLS].apply(lambda r: (r == "Yes").sum(), axis=1)
        X["charges_per_service"] = X["MonthlyCharges"] / (X["service_count"] + 1)
        X["is_new_customer"]     = (X["tenure"] <= 12).astype(int)
        X["is_long_term"]        = (X["tenure"] >= 48).astype(int)
        return X

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def show_batch_prediction(models):
    """Interface for batch predictions on multiple customers."""

    st.markdown("## 📤 Batch Prediction")
    st.markdown(
        "Upload a CSV file with customer data to predict churn for multiple customers at once. "
        "The file should contain the same columns as the training dataset (excluding `customerID` and `Churn`)."
    )

    if not models:
        st.error("No models available for prediction.")
        return

    model_name = st.selectbox("Select Model for Batch Prediction", list(models.keys()))
    uploaded_file = st.file_uploader("📂 Upload Customer CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Drop columns that shouldn't be passed to the model
            id_col = df['customerID'] if 'customerID' in df.columns else None
            label_col = df['Churn'] if 'Churn' in df.columns else None
            predict_df = df.drop(columns=[c for c in ['customerID', 'Churn'] if c in df.columns])

            # Convert TotalCharges to numeric if present
            if 'TotalCharges' in predict_df.columns:
                predict_df['TotalCharges'] = pd.to_numeric(predict_df['TotalCharges'], errors='coerce')

            st.markdown("### 👁 Preview of Uploaded Data")
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Total rows: {len(df)}")

            if st.button("🚀 Run Batch Predictions", key="batch_predict"):
                model = models[model_name]

                # Apply feature engineering BEFORE inference
                predict_df_engineered = _engineer_features(predict_df)

                predictions = model.predict(predict_df_engineered)
                probas = model.predict_proba(predict_df_engineered)[:, 1]

                results_df = predict_df.copy()
                if id_col is not None:
                    results_df.insert(0, 'customerID', id_col.values)
                if label_col is not None:
                    results_df['Actual_Churn'] = label_col.values

                results_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in predictions]
                results_df['Churn_Probability (%)'] = (probas * 100).round(2)

                # Summary stats
                n_churn = sum(predictions)
                n_total = len(predictions)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Customers", n_total)
                c2.metric("Predicted to Churn", n_churn)
                c3.metric("Churn Rate", f"{n_churn/n_total*100:.1f}%")

                st.markdown("### 📋 Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                # Download
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="⬇️ Download Predictions (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
