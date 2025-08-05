import streamlit as st
import pandas as pd
import joblib
import os

# Load model and tools
model = joblib.load('poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🏠 Boston Housing Price Prediction")

st.markdown("""
This app predicts the *median value of a house* (in $1000s) using three features:

- *RM*: Average number of rooms per dwelling  
- *PTRATIO*: Pupil-to-teacher ratio by town  
- *LSTAT*: % of lower status population  

ℹ️ Please enter realistic values:
- RM: 3.0 – 9.0  
- PTRATIO: 12.0 – 22.0  
- LSTAT: 1.0 – 40.0
""")

# User input (with realistic ranges)
st.subheader("🔢 Enter feature values:")
rm = st.number_input("Average number of rooms per dwelling (RM)", min_value=3.0, max_value=9.0)
ptratio = st.number_input("Pupil-teacher ratio by town (PTRATIO)", min_value=12.0, max_value=22.0)
lstat = st.number_input("Percentage of lower status population (LSTAT)", min_value=1.0, max_value=40.0)

# Prediction
if st.button("Predict"):
    # Step 0: Create input DataFrame
    input_df = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
    
    # Step 1: Scale input
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=["RM", "PTRATIO", "LSTAT"])
    
    # Step 2: Polynomial transformation
    input_poly = poly.transform(input_scaled)
    
    # Step 3: Predict
    prediction = model.predict(input_poly)[0]
    
    # Step 4: Display result
    if prediction < 0:
        st.warning("⚠️ Warning: Predicted price is negative. Please check your input values.")
    else:
        st.success(f"💰 Estimated MEDV (Housing Price): ${prediction:.2f}k")