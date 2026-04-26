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

st.set_page_config(
    page_title="Customer Churn Prediction", 
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhance headers */
    h1 {
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #6B7280;
        font-size: 1.125rem;
        margin-bottom: 2rem;
    }
    
    /* Make buttons pop a bit more */
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

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

st.title(":material/analytics: Customer Churn Prediction")
st.markdown('<p class="subtitle">Predict whether a customer is at risk of churning based on their profile and usage.</p>', unsafe_allow_html=True)

if not models:
    st.error("No models found. Please run the training script first.")
    st.stop()

# Navigation sidebar
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Select Page", [":material/monitoring: Metrics", ":material/person: Single Prediction", ":material/group: Batch Prediction"])

if page == ":material/monitoring: Metrics":
    show_metrics()
elif page == ":material/person: Single Prediction":
    show_single_prediction(models)
elif page == ":material/group: Batch Prediction":
    show_batch_prediction(models)
