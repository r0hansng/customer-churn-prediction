import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.metrics import show_metrics
from ui.single_prediction import show_single_prediction
from ui.batch_prediction import show_batch_prediction

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load models safely
@st.cache_resource
def load_models():
    models = {}
    model_dir = "models"
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.joblib'):
                name = f.replace('.joblib', '').replace('_', ' ').title()
                models[name] = joblib.load(os.path.join(model_dir, f))
    return models

models = load_models()

st.title("Customer Churn Prediction")
st.markdown("Predict whether a customer is at risk of churning based on their profile and usage.")

if not models:
    st.error("No models found. Please run the training script first.")
    st.stop()

# Navigation sidebar
page = st.sidebar.radio("Select Page", ["Metrics", "Single Prediction", "Batch Prediction"])

if page == "Metrics":
    show_metrics()
elif page == "Single Prediction":
    show_single_prediction(models)
elif page == "Batch Prediction":
    show_batch_prediction(models)
