import streamlit as st
import pandas as pd
import io

try:
    from src.retention.graph_engine import run_batch_retention_engine
    batch_engine_available = True
except ImportError:
    batch_engine_available = False

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

    st.markdown("## :material/group: Batch Prediction")
    st.markdown(
        "Upload a CSV file with customer data to predict churn for multiple customers at once. "
        "The file should contain the same columns as the training dataset (excluding `customerID` and `Churn`)."
    )

    if not models:
        st.error("No models available for prediction.")
        return

    with st.container(border=True):
        st.markdown("### :material/upload_file: Data Upload & Setup")
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox("Select Model for Batch Prediction", list(models.keys()))
        with col2:
            uploaded_file = st.file_uploader("Upload Customer CSV", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Drop columns that shouldn't be passed to the model
            id_col    = df['customerID'] if 'customerID' in df.columns else None
            label_col = df['Churn']      if 'Churn'       in df.columns else None
            predict_df = df.drop(columns=[c for c in ['customerID', 'Churn'] if c in df.columns])

            # Convert TotalCharges to numeric if present
            if 'TotalCharges' in predict_df.columns:
                predict_df['TotalCharges'] = pd.to_numeric(predict_df['TotalCharges'], errors='coerce')

            with st.expander("Preview Uploaded Data", icon=":material/visibility:", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"Total rows: {len(df)}")

            # ── Run Batch Predictions ─────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn_col1, predict_btn_col2, predict_btn_col3 = st.columns([1, 2, 1])
            with predict_btn_col2:
                run_batch = st.button("Run Batch Predictions", icon=":material/rocket_launch:", key="batch_predict", use_container_width=True)

            if run_batch:
                model = models[model_name]

                # Apply feature engineering BEFORE inference
                predict_df_engineered = _engineer_features(predict_df)

                probas      = model.predict_proba(predict_df_engineered)[:, 1]
                predictions = (probas >= 0.35).astype(int)

                # Persist results in session_state so retention button can access them
                st.session_state["batch_predictions"] = predictions
                st.session_state["batch_probas"]      = probas
                st.session_state["batch_predict_df"]  = predict_df
                st.session_state["batch_id_col"]      = id_col
                st.session_state["batch_label_col"]   = label_col
                # Clear any old strategy when new predictions run
                st.session_state.pop("batch_strategy",    None)
                st.session_state.pop("batch_seg_profile", None)

            # ── Show results if predictions exist in session_state ────────
            if "batch_predictions" in st.session_state:
                st.markdown("---")
                
                predictions = st.session_state["batch_predictions"]
                probas      = st.session_state["batch_probas"]
                predict_df  = st.session_state["batch_predict_df"]
                id_col      = st.session_state["batch_id_col"]
                label_col   = st.session_state["batch_label_col"]

                results_df = predict_df.copy()
                if id_col is not None:
                    results_df.insert(0, 'customerID', id_col.values)
                if label_col is not None:
                    results_df['Actual_Churn'] = label_col.values

                results_df['Predicted_Churn']       = ['Yes' if p == 1 else 'No' for p in predictions]
                results_df['Churn_Probability (%)'] = (probas * 100).round(2)

                n_churn = int(sum(predictions))
                n_total = len(predictions)

                # Summary metrics
                with st.container(border=True):
                    st.markdown("### :material/bar_chart: Prediction Summary")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Customers Evaluated", n_total)
                    c2.metric("Predicted to Churn", n_churn, delta=f"{(n_churn/n_total*100):.1f}%", delta_color="inverse")
                    c3.metric("Safe Customers", n_total - n_churn, delta=f"{((n_total - n_churn)/n_total*100):.1f}%", delta_color="normal")

                st.markdown("### :material/table: Detailed Results")
                
                # Style the dataframe to highlight churners
                def highlight_churn(val):
                    color = 'red' if val == 'Yes' else 'green'
                    return f'color: {color}; font-weight: bold'
                
                styled_df = results_df.style.map(highlight_churn, subset=['Predicted_Churn'])
                
                st.dataframe(styled_df, use_container_width=True, height=300)

                # Download
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                
                dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
                with dl_col2:
                    st.download_button(
                        label="Download Full Predictions (CSV)",
                        icon=":material/download:",
                        data=csv_buffer.getvalue(),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # ── Collective Retention Strategy ─────────────────────────
                if n_churn > 0:
                    st.markdown("---")
                    with st.container(border=True):
                        st.markdown("### :material/psychology: Collective Retention Strategy")

                        if batch_engine_available:
                            st.info(
                                f"**{n_churn} customers** are predicted to churn. "
                                "Instead of individual strategies, we can analyse their shared "
                                "characteristics and generate **one cost-effective retention programme** "
                                "for the entire at-risk segment — via a single AI request.",
                                icon=":material/lightbulb:"
                            )

                            if st.button("Generate Segment Retention Strategy", icon=":material/auto_awesome:", key="batch_retention", use_container_width=True):
                                # Build at-risk subset from session_state data
                                at_risk_mask = predictions == 1
                                at_risk_raw  = predict_df[at_risk_mask].copy()
                                at_risk_prob = probas[at_risk_mask].tolist()

                                with st.spinner(
                                    f"Analysing {n_churn} at-risk customers and generating "
                                    "segment-level strategy via LangGraph + RAG..."
                                ):
                                    try:
                                        strategy, seg_profile = run_batch_retention_engine(
                                            at_risk_raw, at_risk_prob
                                        )
                                        st.session_state["batch_strategy"]    = strategy
                                        st.session_state["batch_seg_profile"] = seg_profile
                                    except Exception as ex:
                                        st.error(f"Strategy generation failed: {ex}")
                        else:
                            st.warning("Retention engine not available — check GOOGLE_API_KEY in Streamlit Secrets.", icon=":material/warning:")

                # Render strategy if it exists in session_state
                if "batch_strategy" in st.session_state:
                    seg = st.session_state.get("batch_seg_profile", {})

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### :material/monitoring: At-Risk Segment Profile")
                    
                    # Modern profile metrics
                    k1, k2, k3, k4, k5 = st.columns(5)
                    with k1: st.metric("At-Risk Count", seg.get("total_at_risk_customers", "—"))
                    with k2: st.metric("Avg Risk", f"{float(seg.get('avg_churn_probability', 0)):.1%}" if str(seg.get('avg_churn_probability')).replace('.','',1).isdigit() else "—")
                    with k3: st.metric("Avg Tenure", f"{seg.get('avg_tenure_months','—')} mo")
                    with k4: st.metric("Avg Bill", f"${seg.get('avg_monthly_charges', '—')}")
                    with k5: st.metric("Active Services", seg.get("avg_active_services", "—"))

                    col_a, col_b, col_c = st.columns(3)
                    with col_a: st.metric("Dominant Contract", seg.get("most_common_contract", "—"))
                    with col_b: st.metric("Dominant Internet", seg.get("most_common_internet_service", "—"))
                    with col_c: st.metric("Dominant Payment",  seg.get("most_common_payment_method", "—"))

                    st.markdown("---")
                    st.markdown("#### :material/description: Recommended Retention Programme")
                    st.info(st.session_state["batch_strategy"], icon=":material/assignment:")

        except Exception as e:
            st.error(f"Error processing file: {e}")
