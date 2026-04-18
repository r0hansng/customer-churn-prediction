import streamlit as st
import pandas as pd
import numpy as np
import os

def show_single_prediction(models):
    """Interface for making single customer predictions."""

    st.markdown("## 🎯 Single Customer Prediction")

    if not models:
        st.error("No models available for prediction.")
        return

    st.markdown("Enter customer information below to predict churn risk:")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges) or 1000.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = 1 if st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes" else 0

    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.markdown("### Service & Contract Details")
    col3, col4 = st.columns(2)

    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col4:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    if st.button("🔍 Predict Churn Risk", key="single_predict"):
        # Build the input DataFrame matching the training feature schema
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        st.markdown("### 📊 Prediction Results")

        for model_name, model in models.items():
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                if prediction == 1:
                    st.error(f"**{model_name}**: 🔴 WILL CHURN — {proba[1]*100:.1f}% probability")
                else:
                    st.success(f"**{model_name}**: 🟢 WON'T CHURN — {proba[1]*100:.1f}% churn probability")
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
