import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
The app will also explain **why** it made its prediction!
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
    # Group inputs into a dataframe 
    feature_names = ['fixed_acidity', 'residual_sugar', 'free_sulfur_dioxide', 'density', 'pH']
    display_names = ['Fixed Acidity', 'Residual Sugar', 'Free Sulfur Dioxide', 'Density', 'pH']
    
    input_data = pd.DataFrame([[
        fixed_acidity, residual_sugar, free_sulfur_dioxide, density, pH
    ]], columns=feature_names)
    
    try:
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        st.markdown("---")
        
        # Display Prediction Result
        if prediction[0] == 1:
            st.success("🌟 **Prediction: Good/Premium Quality Wine!**")
        else:
            st.error("🍷 **Prediction: Standard/Poor Quality Wine**")
            
        st.markdown("---")
        
        # ==========================================
        # EXPLAINABILITY SECTION
        # ==========================================
        st.subheader("🔍 Why did the model make this prediction?")
        st.write("This chart shows how much each property contributed to this specific prediction. **Green bars** push the prediction towards Premium, while **Red bars** pull it towards Standard.")
        
        # Calculate how much each feature contributed to this specific decision
        # Contribution = (Model Coefficient) * (Scaled Input Value)
        contributions = model.coef_[0] * input_scaled[0]
        
        # Create a DataFrame for the plot
        contrib_df = pd.DataFrame({
            'Feature': display_names,
            'Contribution': contributions
        }).sort_values(by='Contribution', ascending=True)
        
        # Plotting the feature contributions
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Color positive contributions green, negative red
        colors = ['#ff9999' if x < 0 else '#99ff99' for x in contrib_df['Contribution']]
        
        # Create horizontal bar chart
        ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors, edgecolor='black')
        
        # Format the chart
        ax.set_xlabel("Impact on Prediction (Log-Odds)")
        ax.set_title("Feature Contributions for this Wine")
        ax.axvline(0, color='black', linewidth=1) # Add a vertical line at 0
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
