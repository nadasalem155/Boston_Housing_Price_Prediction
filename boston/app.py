import streamlit as st
import pandas as pd
import joblib

model = joblib.load('poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Boston Housing Price Prediction")

# User input
st.subheader("Enter feature values:")
rm = st.number_input("Average number of rooms per dwelling (RM)", value=0.0)
ptratio = st.number_input("Pupil-teacher ratio by town (PTRATIO)", value=0.0)
lstat = st.number_input("Percentage of lower status population (LSTAT)", value=0.0)

if st.button("Predict"):
    # Step 0: Create a DataFrame with proper column names
    input_df = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
    
    # Step 1: Scale the input and preserve column names
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=["RM", "PTRATIO", "LSTAT"])

    # Step 2: Apply the polynomial transformation
    input_poly = poly.transform(input_scaled)
    
    # Step 3: Predict
    prediction = model.predict(input_poly)[0]
    
    # Step 4: Display result
    st.success(f"Estimated MEDV (Housing Price): ${prediction:.2f}")