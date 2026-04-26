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

try:
    from src.preprocessing.preprocess import _engineer_features
except ImportError:
    # Fallback: define inline so the UI never crashes even if src path differs
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

def show_single_prediction(models):
    """Interface for making single customer predictions."""

    st.markdown("## :material/person: Single Customer Prediction")

    if not models:
        st.error("No models available for prediction.")
        return

    st.markdown("Enter customer information below to predict churn risk:")

    with st.container(border=True):
        st.markdown("### :material/account_circle: Demographics & Account")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = 1 if st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes" else 0
            partner = st.selectbox("Partner", ["Yes", "No"])

        with col2:
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges) or 1000.0)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

    with st.container(border=True):
        st.markdown("### :material/settings: Services Subscribed")
        col4, col5, col6 = st.columns(3)

        with col4:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col5:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        with col6:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button("Predict Churn Risk", icon=":material/search:", key="single_predict", use_container_width=True)

    if predict_button:
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

        # Apply feature engineering BEFORE passing to models
        # (models were trained with engineered features pre-computed)
        input_data_engineered = _engineer_features(input_data)

        st.markdown("---")
        st.markdown("### :material/bar_chart: Prediction Results")

        # Collect all model predictions first
        all_results = {}
        for model_name, model in models.items():
            try:
                proba = model.predict_proba(input_data_engineered)[0]
                prediction = 1 if proba[1] >= 0.35 else 0
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
            st.error(f"### :material/cancel: **Ensemble Verdict: HIGH CHURN RISK**\n**{votes_churn}/{total_models}** models predict churn · **Avg probability: {avg_prob:.1f}%**", icon=":material/warning:")
        else:
            st.success(f"### :material/check_circle: **Ensemble Verdict: LOW CHURN RISK**\n**{votes_churn}/{total_models}** models predict churn · **Avg probability: {avg_prob:.1f}%**", icon=":material/check_circle:")

        st.markdown("#### Individual Model Breakdown")
        # Display in rows of 3 using containers
        model_items = list(all_results.items())
        for row_start in range(0, len(model_items), 3):
            row_items = model_items[row_start:row_start + 3]
            cols = st.columns(len(row_items))
            for col, (model_name, result) in zip(cols, row_items):
                with col:
                    with st.container(border=True):
                        if "error" in result:
                            st.error(f"**{model_name}**\nError: {result['error']}")
                        else:
                            if result["prediction"] == 1:
                                st.markdown(f"**{model_name}**")
                                st.markdown(":material/cancel: **WILL CHURN**")
                            else:
                                st.markdown(f"**{model_name}**")
                                st.markdown(":material/check_circle: **WON'T CHURN**")
                            
                            st.metric("Churn Prob.", f"{result['churn_prob']:.1f}%")

        # Determine if overall risk is high based on Ensemble Verdict (majority votes)
        if votes_churn > total_models / 2:
            st.session_state['high_risk_customer'] = True
            st.session_state['last_input_data'] = input_data
            st.session_state['last_churn_prob'] = avg_prob / 100
        else:
            st.session_state['high_risk_customer'] = False
            if 'retention_strategy' in st.session_state:
                del st.session_state['retention_strategy']

    # Display Strategy Button if high risk
    if st.session_state.get('high_risk_customer', False) and engine_available:
        st.markdown("---")
        with st.container(border=True):
            st.warning("#### High Risk Customer Detected", icon=":material/warning:")
            st.markdown("This customer has a high likelihood of churning. Generate a personalized retention strategy below.")
            
            if st.button("Generate AI Retention Strategy", icon=":material/auto_awesome:", use_container_width=True):
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
            st.markdown("### :material/description: Recommended Retention Strategy")
            st.info(st.session_state['retention_strategy'], icon=":material/lightbulb:")
