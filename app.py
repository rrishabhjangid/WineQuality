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
Enter the physiochemical properties of the white wine below to predict if it is of **Premium Quality** (Score > 7) or **Standard Quality**.
""")

# Create layout with columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=6.8, step=0.1)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=70.0, value=6.4, step=0.1)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=500.0, value=138.0, step=1.0)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.49, step=0.01)

with col2:
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.28, step=0.01)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.045, step=0.001)
    density = st.number_input("Density", min_value=0.9, max_value=1.1, value=0.994, step=0.001)
    alcohol = st.number_input("Alcohol (%)", min_value=5.0, max_value=20.0, value=10.5, step=0.1)

with col3:
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=2.0, value=0.33, step=0.01)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=300.0, value=35.0, step=1.0)
    pH = st.number_input("pH", min_value=0.0, max_value=5.0, value=3.19, step=0.01)

# Predict button
if st.button("Predict Wine Quality"):
    # Group inputs into a dataframe (matching the training data columns)
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]], columns=[
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
        'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ])
    
    # Scale the features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🌟 **Prediction: Premium Quality Wine!** (Quality Score > 7)")
    else:
        st.info("🍷 **Prediction: Standard Quality Wine** (Quality Score <= 7)")
