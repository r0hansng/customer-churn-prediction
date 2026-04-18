import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from src.retention.graph_engine import run_retention_engine
    engine_available = True
except ImportError as e:
    engine_available = False
    print(f"Engine not available: {e}")
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

        # Collect all model predictions first
        all_results = {}
        for model_name, model in models.items():
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                all_results[model_name] = {
                    "prediction": prediction,
                    "churn_prob": proba[1] * 100
                }
            except Exception as e:
                all_results[model_name] = {"error": str(e)}

        # Ensemble verdict banner
        votes_churn = sum(1 for r in all_results.values() if r.get("prediction") == 1)
        avg_prob = sum(r["churn_prob"] for r in all_results.values() if "churn_prob" in r) / max(len(all_results), 1)
        total_models = len(all_results)

        if votes_churn > total_models / 2:
            st.error(f"🔴 **Ensemble Verdict: HIGH CHURN RISK** — {votes_churn}/{total_models} models predict churn · Avg probability: {avg_prob:.1f}%")
        else:
            st.success(f"🟢 **Ensemble Verdict: LOW CHURN RISK** — {votes_churn}/{total_models} models predict churn · Avg probability: {avg_prob:.1f}%")

        # Display in rows of 3
        model_items = list(all_results.items())
        for row_start in range(0, len(model_items), 3):
            row_items = model_items[row_start:row_start + 3]
            cols = st.columns(len(row_items))
            for col, (model_name, result) in zip(cols, row_items):
                with col:
                    if "error" in result:
                        st.error(f"**{model_name}**\nError: {result['error']}")
                    else:
                        if result["prediction"] == 1:
                            st.error(f"**{model_name}**\n\n🔴 **WILL CHURN**")
                        else:
                            st.success(f"**{model_name}**\n\n🟢 **WON'T CHURN**")
                        st.metric("Churn Probability",     f"{result['churn_prob']:.1f}%")
                        st.metric("Retention Probability", f"{100 - result['churn_prob']:.1f}%")

        # Determine if overall risk is high (for RAG engine trigger)
        max_churn_prob = max(
            (r["churn_prob"] / 100 for r in all_results.values() if "churn_prob" in r),
            default=0
        )
                
        if max_churn_prob > 0.5:
            st.session_state['high_risk_customer'] = True
            st.session_state['last_input_data'] = input_data
            st.session_state['last_churn_prob'] = max_churn_prob
        else:
            st.session_state['high_risk_customer'] = False
            if 'retention_strategy' in st.session_state:
                del st.session_state['retention_strategy']

    # Display Strategy Button if high risk
    if st.session_state.get('high_risk_customer', False) and engine_available:
        st.markdown("---")
        st.warning("⚠️ High Risk Customer Detected.")
        if st.button("✨ Generate Retention Strategy (AI)"):
            with st.spinner("Analyzing policies and generating strategy using LangGraph & RAG..."):
                try:
                    strategy = run_retention_engine(
                        st.session_state['last_input_data'], 
                        st.session_state['last_churn_prob']
                    )
                    st.session_state['retention_strategy'] = strategy
                except Exception as e:
                    st.error(f"Error generating strategy: {str(e)}")
                    
    if 'retention_strategy' in st.session_state:
        st.markdown("### 📝 Recommended Retention Strategy")
        st.info(st.session_state['retention_strategy'])
