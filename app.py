import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('wine_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Set up the Streamlit app title and description
st.title("🍷 White Wine Quality Predictor")
st.write("""
Enter the physiochemical properties of the white wine below to predict its quality.
""")

# Create layout with columns for the 5 remaining inputs
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=6.8, step=0.1)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=300.0, value=35.0, step=1.0)
    pH = st.number_input("pH", min_value=0.0, max_value=5.0, value=3.19, step=0.01)

with col2:
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=70.0, value=6.4, step=0.1)
    density = st.number_input("Density", min_value=0.900, max_value=1.100, value=0.994, step=0.001)

# Predict button
if st.button("Predict Wine Quality"):
    # Group inputs into a dataframe with exact column names and order from training
    input_data = pd.DataFrame([[
        fixed_acidity, residual_sugar, free_sulfur_dioxide, density, pH
    ]], columns=[
        'fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'density', 'pH'
    ])
    
    try:
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        st.markdown("---")
        # Assuming 1 is Premium/Good and 0 is Standard/Bad based on typical binary mappings
        if prediction[0] == 1:
            st.success("🌟 **Prediction: Good/Premium Quality Wine!**")
        else:
            st.info("🍷 **Prediction: Standard/Poor Quality Wine**")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
