import streamlit as st
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load('poly_model.pkl')
scaler = joblib.load('scaler.pkl')
poly_transformer = joblib.load('poly_transformer.pkl')

st.title("Boston Housing Price Prediction (Simplified)")
st.write("Enter details below to predict house price (in $1000s)")

# User inputs (3 features only)
rm = st.number_input("RM - Average number of rooms per dwelling", min_value=0.0)
ptratio = st.number_input("PTRATIO - Pupil-teacher ratio by town", min_value=0.0)
lstat = st.number_input("LSTAT - % lower status of the population", min_value=0.0)

if st.button("Predict Price"):
    # Prepare input for prediction
    input_data = np.array([[rm, ptratio, lstat]])
    input_scaled = scaler.transform(input_data)
    input_poly = poly_transformer.transform(input_scaled)
    
    prediction = model.predict(input_poly)
    
    st.success(f"Predicted Median House Price: ${prediction[0]:.2f}K")