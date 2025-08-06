import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model and preprocessors
regressor = joblib.load("poly_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly_transformer.pkl")

st.set_page_config(page_title="Boston House Price App", layout="wide")

# Sidebar with explanations
with st.sidebar:
    st.header("Feature Explanation")
    
    st.markdown("""
    ### RM (Average number of rooms per dwelling)
    RM represents the average number of rooms in a house.  
    Generally, the more rooms, the higher the house value.  
    However, due to the polynomial regression model, this relationship isn't always linear – sometimes increasing RM may lead to a decrease in predicted price.
    """)

    st.markdown("""
    ### PTRATIO (Pupil-Teacher Ratio)
    PTRATIO is the number of students per teacher in schools within the area.  
    Lower PTRATIO usually indicates better education quality.  
    As a result, areas with lower PTRATIO often have higher house prices.
    """)

    st.markdown("""
    ### LSTAT (% lower status population)
    LSTAT measures the percentage of people with lower socio-economic status in a neighborhood.  
    Higher LSTAT often correlates with lower housing demand and thus lower prices.  
    So, as LSTAT increases, predicted price tends to decrease.
    """)

    st.markdown("""
    ℹ️ Note:  
    This model uses polynomial regression, meaning relationships between features and the target (MEDV) may not always be linear.  
    In some ranges, increasing a feature may decrease the predicted value depending on the curve's shape.
    """)

# ===== Title and Input Sliders =====
st.title("🏠 Boston House Price Predictor")
st.markdown("Use the sliders below to explore how different features affect the predicted house price.")

# Sliders with minimum values set correctly
col1, col2 = st.columns(2)
with col1:
    rm = st.slider("RM (Average Rooms per Dwelling)", 3.56, 8.78,6.17 , 0.1)
with col2:
    ptratio = st.slider("PTRATIO (Pupil-Teacher Ratio)", 12.6, 22.0,17.3 , 0.1)

lstat = st.slider("LSTAT (% Lower Status Population)", 1.73, 37.97,19.85 , 0.5)

# Prediction
input_df = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
input_scaled = scaler.transform(input_df)
input_poly = poly.transform(input_scaled)
predicted_price = regressor.predict(input_poly)[0]

st.markdown(f"### 💰 Predicted House Price: ${predicted_price * 1000:.2f}")

# ===== Feature vs MEDV Plots =====
st.markdown("## 📊 Feature Impact on Price (MEDV)")

def generate_plot(feature_name, feature_values, size=(2.2, 1.5)):
    preds = []
    for val in feature_values:
        row = pd.DataFrame([[rm, ptratio, lstat]], columns=["RM", "PTRATIO", "LSTAT"])
        row[feature_name] = val
        scaled = scaler.transform(row)
        poly_features = poly.transform(scaled)
        preds.append(regressor.predict(poly_features)[0])
    
    fig, ax = plt.subplots(figsize=size)
    ax.plot(feature_values, preds)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("MEDV")
    ax.set_title(f"{feature_name} vs MEDV", fontsize=8)
    ax.grid(True)

    # Show only 3 ticks on both axes
    ax.set_xticks(np.linspace(feature_values.min(), feature_values.max(), 3))
    ax.set_yticks(np.linspace(min(preds), max(preds), 3))

    return fig

# Arrange plots in one row
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    st.markdown("RM")
    st.pyplot(generate_plot("RM", np.linspace(3.56, 8.78, 50)))

with row1_col2:
    st.markdown("PTRATIO")
    st.pyplot(generate_plot("PTRATIO", np.linspace(12.6, 22.0, 50)))

with row1_col3:
    st.markdown("LSTAT")
    st.pyplot(generate_plot("LSTAT", np.linspace(1.73, 37.97, 50)))