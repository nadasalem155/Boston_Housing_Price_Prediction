import streamlit as st
import pandas as pd
import joblib

# Load the trained model and polynomial transformer
model = joblib.load("poly_model.pkl")
poly = joblib.load("poly_transformer.pkl")  # Make sure you saved it earlier

# App title
st.title("Boston Housing Price Prediction")

# User input
st.subheader("Enter feature values:")
rm = st.number_input("Average number of rooms per dwelling (RM)", value=6.0)
ptratio = st.number_input("Pupil-teacher ratio by town (PTRATIO)", value=18.0)
lstat = st.number_input("Percentage of lower status population (LSTAT)", value=12.5)

# When the user clicks Predict
if st.button("Predict"):
    # Create a DataFrame with proper column names
    input_df = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
    
    # Apply the same polynomial transformation
    input_poly = poly.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_poly)[0]
    
    # Display result
    st.success(f"Estimated MEDV (Housing Price): ${prediction:.2f}")
