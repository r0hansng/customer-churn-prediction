import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Model Evaluation Metrics ---
with st.expander("📊 View Model Evaluation Metrics"):
    st.markdown("""
    These are the evaluation metrics for the models trained on the historical dataset:
    
    | Model | Accuracy | Precision | Recall | F1 Score |
    | :--- | :--- | :--- | :--- | :--- |
    | **Logistic Regression** | 82.11% | 68.62% | 59.79% | 63.90% |
    | **Decision Tree** | 71.26% | 45.98% | 49.06% | 47.47% |
    | **MLP (Neural Net)** | 79.21% | 63.51% | 50.40% | 56.20% |
    
    *Logistic Regression performed the best overall on this dataset.*
    """)

# --- Prediction Mode Selection ---
prediction_mode = st.radio("Select Prediction Mode", ["Single Customer (Manual Input)", "Batch Prediction (CSV Upload)"], horizontal=True)

st.divider()

selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))

if prediction_mode == "Single Customer (Manual Input)":
    # Sidebar for manual inputs
    st.sidebar.header("Customer Profile")
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["0", "1"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': int(senior_citizen),
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])
    
    st.subheader("Selected Customer Data")
    st.dataframe(input_data)
    
    if st.button("Predict Churn", type="primary"):
        model = models[selected_model_name]
        try:
            prob = model.predict_proba(input_data)[0][1]
            pred = model.predict(input_data)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{prob:.2%}")
            
            with col2:
                if pred == 1:
                    st.error("🚨 HIGH RISK of Churn")
                else:
                    st.success("✅ LOW RISK of Churn")
            
            st.info("Insights: Higher monthly charges and month-to-month contracts strongly drive churn (based on model coefficients).")
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

else:
    # CSV Upload Mode
    st.subheader("Upload Customer Data")
    st.markdown("Upload a CSV file with the same format as the training data to get batch predictions.")
    
    # Initialize session state for batch results
    if 'batch_results' not in st.session_state:
        st.session_state['batch_results'] = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state['uploaded_filename'] = None
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Reset session state if a new file is uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state['uploaded_filename']:
        st.session_state['batch_results'] = None
        st.session_state['uploaded_filename'] = uploaded_file.name
        
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"Preview of uploaded data (`{uploaded_file.name}`):")
            st.dataframe(batch_data.head())
            
            if st.button("Run Batch Prediction", type="primary"):
                model = models[selected_model_name]
                
                # Preprocess TotalCharges correctly for new data like during training
                if 'TotalCharges' in batch_data.columns:
                    batch_data['TotalCharges'] = pd.to_numeric(batch_data['TotalCharges'], errors='coerce')
                
                # Handle customerID column if present
                display_df = batch_data.copy()
                predict_df = batch_data.copy()
                
                if 'customerID' in predict_df.columns:
                    predict_df = predict_df.drop('customerID', axis=1)
                
                # Remove target column if it accidentally exists in the upload
                if 'Churn' in predict_df.columns:
                    predict_df = predict_df.drop('Churn', axis=1)
                    
                with st.spinner('Predicting churn risk...'):
                    predictions = model.predict(predict_df)
                    probabilities = model.predict_proba(predict_df)[:, 1]
                    
                    # Keep raw probabilities for plotting
                    display_df['Churn_Probability_Raw'] = probabilities
                    
                    display_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                    display_df['Churn_Probability'] = [f"{prob:.2%}" for prob in probabilities]
                    
                    # Store results in session state
                    st.session_state['batch_results'] = display_df
                    
        except Exception as e:
            st.error(f"Error processing the file. Make sure it matches the training data format. Details: {str(e)}")

    # Display results if they exist in session state (works even if uploaded_file gets cleared from UI)
    if st.session_state['batch_results'] is not None:
        display_df = st.session_state['batch_results']
        
        filename_msg = f" from `{st.session_state['uploaded_filename']}`" if st.session_state['uploaded_filename'] else ""
        st.success(f"Successfully predicted churn for {len(display_df)} customers{filename_msg}!")
        
        st.markdown("---")
        st.subheader("📊 Batch Prediction Dashboard")
        
        churn_count = sum(display_df['Churn_Prediction'] == 'Yes')
        retain_count = sum(display_df['Churn_Prediction'] == 'No')
        
        # Top Metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Total Customers Processed", len(display_df))
        metrics_col2.metric("Predicted to Churn 🚨", churn_count, f"{churn_count/len(display_df):.1%} of total", delta_color="inverse")
        metrics_col3.metric("Predicted to Stay ✅", retain_count, f"{retain_count/len(display_df):.1%} of total")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Churn Proportion**")
            churn_dist = pd.DataFrame({
                'Status': ['Churn', 'Retain'],
                'Count': [churn_count, retain_count]
            })
            st.bar_chart(churn_dist.set_index('Status'), color="#ff4b4b")
            
        with chart_col2:
            st.markdown("**Churn Probability Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(display_df['Churn_Probability_Raw'], bins=20, kde=True, ax=ax, color='#1f77b4')
            ax.set_xlabel('Churn Probability')
            ax.set_ylabel('Number of Customers')
            ax.set_xlim(0, 1)
            # Remove top and right spines
            sns.despine(ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("📋 Detailed Customer List")
        
        # Actionable data table
        filter_option = st.radio("Filter customer list:", ["View All Customers", "View Only At-Risk Customers (Churn = Yes)"], horizontal=True)
        
        view_df = display_df.copy()
        if filter_option == "View Only At-Risk Customers (Churn = Yes)":
            view_df = view_df[view_df['Churn_Prediction'] == 'Yes']
            
        # Drop the raw probability column before displaying to user
        view_df_display = view_df.drop(columns=['Churn_Probability_Raw'])
        
        st.dataframe(view_df_display, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download button
        csv = view_df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Shown Predictions as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
            use_container_width=True
        )
