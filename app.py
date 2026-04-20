import streamlit as st
import numpy as np
import pickle

# Load model + scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🍷 Wine Quality Predictor")

st.write("Enter wine chemical properties to predict quality")

# Inputs (based on final features)
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 5.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 30.0)
density = st.number_input("Density", 0.990, 1.010, 0.995)
pH = st.number_input("pH", 2.5, 4.5, 3.2)

# Predict
if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, residual_sugar, free_sulfur_dioxide, density, pH]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("⚠️ Poor Quality Wine")
