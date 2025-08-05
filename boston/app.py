import streamlit as st
import pandas as pd
import joblib
import os

# Get the current working directory
current_dir = os.getcwd()

# Paths to the saved model, scaler, and polynomial transformer
model_path = os.path.join(current_dir, 'poly_model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')
poly_path = os.path.join(current_dir, 'poly_transformer.pkl')

# Load the model, scaler, and polynomial features transformer
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
poly = joblib.load(poly_path)

# App title
st.title('Boston Housing Price Prediction')

# Description
st.write('Enter the following values to predict the house price:')

# User inputs
rm = st.number_input('Average number of rooms (RM)', min_value=0.0, value=6.0, step=0.1)
ptratio = st.number_input('Pupil-teacher ratio by town (PTRATIO)', min_value=0.0, value=15.0, step=0.1)
lstat = st.number_input('Percentage of lower status population (LSTAT)', min_value=0.0, value=10.0, step=0.1)

# Prediction button
if st.button('Predict Price'):
    # Create input DataFrame
    input_data = pd.DataFrame([[rm, ptratio, lstat]], columns=['RM', 'PTRATIO', 'LSTAT'])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Transform the input data using PolynomialFeatures
    poly_input = poly.transform(scaled_input)

    # Make prediction
    prediction = model.predict(poly_input)[0]

    # Display the result
    st.success(f'Estimated house price: ${prediction:.2f}')