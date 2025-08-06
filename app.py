import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and preprocessors
model = joblib.load("poly_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly_transformer.pkl")  # ✅ Correct file name

# Page config
st.set_page_config(page_title="Boston House Price App", layout="centered")

# Title and instruction
st.title("🏠 Boston House Price Predictor")
st.markdown("Adjust the inputs below to estimate the predicted house price.")

# --- Input sliders ---
st.subheader("📊 Input Features")

col1, col2 = st.columns(2)
with col1:
    rm = st.slider("RM (average rooms per dwelling)", 3.0, 10.0, 6.0, 0.1)
with col2:
    ptratio = st.slider("PTRATIO (pupil-teacher ratio)", 10.0, 25.0, 18.0, 0.1)

lstat = st.slider("LSTAT (% lower status population)", 1.0, 40.0, 12.0, 0.5)

# --- Prediction ---
input_data = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
input_scaled = scaler.transform(input_data)
input_poly = poly.transform(input_scaled)
prediction = model.predict(input_poly)[0]

st.markdown(f"### 💰 Predicted House Price: ${prediction * 1000:.2f}")

# --- Function to plot feature vs price ---
def plot_feature_vs_price(feature_name, feature_range):
    preds = []
    for val in feature_range:
        temp = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
        temp[feature_name] = val
        temp_scaled = scaler.transform(temp)
        temp_poly = poly.transform(temp_scaled)
        preds.append(model.predict(temp_poly)[0])
    
    fig, ax = plt.subplots(figsize=(4, 2))  # ✅ Smaller plot
    ax.plot(feature_range, preds)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("MEDV")
    ax.set_title(f"{feature_name} vs MEDV", fontsize=10)
    ax.grid(True)
    return fig

# --- Mini plots for features ---
st.markdown("### 🔍 Feature Impact")

col1, col2 = st.columns(2)
with col1:
    st.markdown("*RM*")
    st.pyplot(plot_feature_vs_price("RM", np.linspace(3, 10, 50)))

with col2:
    st.markdown("*PTRATIO*")
    st.pyplot(plot_feature_vs_price("PTRATIO", np.linspace(10, 25, 50)))

st.markdown("*LSTAT*")
st.pyplot(plot_feature_vs_price("LSTAT", np.linspace(1, 40, 50)))