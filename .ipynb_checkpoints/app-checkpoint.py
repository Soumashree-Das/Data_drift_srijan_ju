import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model/model.pkl")

st.title("🚦 Accident Severity Prediction Dashboard")

st.sidebar.header("Input Features")

# Replace with your real input fields
feature1 = st.sidebar.slider("Feature 1", 0.0, 10.0, 5.0)
feature2 = st.sidebar.slider("Feature 2", 0.0, 10.0, 5.0)
feature3 = st.sidebar.slider("Feature 3", 0.0, 10.0, 5.0)
feature4 = st.sidebar.slider("Feature 4", 0.0, 10.0, 5.0)

input_data = np.array([[feature1, feature2, feature3, feature4]])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Severity Class: {prediction[0]}")
