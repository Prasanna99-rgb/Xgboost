import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("XGBR.pkl", "rb"))

st.set_page_config(page_title="XGBoost Prediction App", layout="centered")

st.title("🔮 XGBoost Regression Prediction App")
st.write("Enter the feature values to get prediction")

# Input fields (8 features)
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")
f5 = st.number_input("Feature 5")
f6 = st.number_input("Feature 6")
f7 = st.number_input("Feature 7")
f8 = st.number_input("Feature 8")

# Prediction button
if st.button("Predict"):

    features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])

    prediction = model.predict(features)

    st.success(f"Prediction Result: {prediction[0]}")
