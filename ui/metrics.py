import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_metrics():
    """Display model evaluation metrics and visualizations."""
    
    st.markdown("## 📊 View Model Evaluation Metrics")
    st.markdown("""
    These metrics were computed by evaluating each saved model on the **held-out test set** (20% of the dataset, 1,409 samples):
    """)

    metrics_data = {
        "Model": ["Logistic Regression", "Decision Tree", "MLP (Neural Net)"],
        "Accuracy":  ["81.97%", "79.99%", "78.57%"],
        "Precision": ["68.42%", "61.88%", "61.41%"],
        "Recall":    ["59.25%", "63.54%", "51.21%"],
        "F1 Score":  ["63.51%", "62.70%", "55.85%"],
    }
    metrics_df = pd.DataFrame(metrics_data).set_index("Model")
    st.table(metrics_df)

    st.markdown("**Confusion Matrices** (rows = Actual, cols = Predicted):")
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col1:
        st.markdown("**Logistic Regression**")
    with cm_col2:
        st.markdown("**Decision Tree**")
    with cm_col3:
        st.markdown("**MLP (Neural Net)**")
