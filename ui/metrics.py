import streamlit as st
import pandas as pd

def show_metrics():
    """Display model evaluation metrics and visualizations."""

    st.markdown("## :material/query_stats: Model Evaluation Metrics")
    
    with st.container(border=True):
        st.markdown("""
        Metrics computed on the **held-out test set** (20% of dataset — 1,409 samples,
        stratified by churn label). Models were selected via **5-fold RandomizedSearchCV**
        optimising **ROC-AUC** on the training set.
        """)
        st.info("""
        **Key Improvements Applied:**
        - `class_weight='balanced'` to penalise missing churners
        - **SMOTE** (Synthetic Minority Oversampling) inside the cross-validation loop
        - **5 engineered features**: avg monthly charge, service count, charges per service, new-customer flag, long-term flag
        - **5 model families** evaluated including Random Forest and Gradient Boosting ensembles
        """, icon=":material/rocket_launch:")

    # Tabs for better organization
    tab1, tab2, tab3 = st.tabs([":material/settings: Best Hyperparameters", ":material/trending_up: Test Set Performance", ":material/tune: Threshold Tuning"])

    with tab1:
        st.markdown("### Best Hyperparameters (RandomizedSearchCV, scoring=ROC-AUC)")
        hp_data = {
            "Model": [
                "Logistic Regression", "Decision Tree", "MLP (Neural Net)",
                "Random Forest", "Gradient Boosting",
            ],
            "Best Parameters": [
                "C=1.0, solver=lbfgs, class_weight=balanced",
                "max_depth=5, min_samples_leaf=5, class_weight=None",
                "hidden_layer_sizes=(50,), alpha=0.01, lr_init=0.001",
                "n_estimators=300, max_depth=10, min_samples_leaf=10, class_weight=None",
                "n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8",
            ],
            "CV ROC-AUC": ["84.73%", "81.60%", "79.61%", "84.47%", "84.59%"],
        }
        st.dataframe(pd.DataFrame(hp_data).set_index("Model"), use_container_width=True)

    with tab2:
        st.markdown("### Test Set Performance (decision threshold = 0.50)")
        metrics_data = {
            "Model": [
                "Logistic Regression", "Decision Tree", "MLP (Neural Net)",
                "Random Forest", "Gradient Boosting",
            ],
            "Accuracy":  ["74.24%", "75.30%", "73.46%", "76.65%", "76.72%"],
            "Precision": ["50.95%", "52.61%", "50.00%", "54.56%", "54.94%"],
            "Recall":    ["78.88%", "70.05%", "65.78%", "71.93%", "68.45%"],
            "F1 Score":  ["61.91%", "60.09%", "56.81%", "62.05%", "60.95%"],
            "ROC-AUC":   ["84.37%", "80.35%", "79.39%", "83.88%", "84.18%"],
        }
        
        # Style dataframe
        df_metrics = pd.DataFrame(metrics_data).set_index("Model")
        st.dataframe(df_metrics, use_container_width=True)

    with tab3:
        st.markdown("### Threshold Tuning (0.50 vs 0.35) — Churn Recall Improvement")
        st.caption(
            "Lowering the decision threshold to 0.35 trades some precision for "
            "significantly higher recall — catching more at-risk customers."
        )
        threshold_data = {
            "Model": [
                "Logistic Regression", "Decision Tree", "MLP (Neural Net)",
                "Random Forest", "Gradient Boosting",
            ],
            "Recall @0.50": ["78.88%", "70.05%", "65.78%", "71.93%", "68.45%"],
            "Recall @0.35": ["89.57%", "81.28%", "74.60%", "84.49%", "82.35%"],
            "F1 @0.50":     ["61.91%", "60.09%", "56.81%", "62.05%", "60.95%"],
            "F1 @0.35":     ["60.04%", "58.02%", "57.94%", "61.30%", "61.72%"],
        }
        st.dataframe(pd.DataFrame(threshold_data).set_index("Model"), use_container_width=True)

    # Confusion matrices at 0.50
    st.markdown("---")
    st.markdown("### :material/grid_view: Confusion Matrices (threshold = 0.50)")
    st.caption("Test set: 1,035 No-Churn / 374 Churn · Rows = Actual · Columns = Predicted")

    # Use bordered containers for confusion matrices
    col1, col2, col3 = st.columns(3)
    
    def render_cm(title, data):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(
                data,
                index=["Actual: No", "Actual: Yes"],
                columns=["Pred: No", "Pred: Yes"],
            ), use_container_width=True)

    with col1:
        render_cm("Logistic Regression", [[751, 284], [79, 295]])

    with col2:
        render_cm("Decision Tree", [[799, 236], [112, 262]])

    with col3:
        render_cm("MLP (Neural Net)", [[789, 246], [128, 246]])

    col4, col5 = st.columns(2)
    with col4:
        render_cm("Random Forest", [[811, 224], [105, 269]])

    with col5:
        render_cm("Gradient Boosting", [[825, 210], [118, 256]])
