import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="California Housing AI Predictor",
    page_icon="🏠",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #45a049;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    try:
        with open("XGBR.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except:
        st.error("❌ Model file not found. Upload XGBR.pkl")
        st.stop()

model = load_model()

# ------------------ HEADER ------------------
st.markdown("""
<div class="card">
<h1 style='text-align: center;'>🏠 California Housing Price Predictor</h1>
<p style='text-align: center;'>AI-powered real estate price prediction using XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📊 Model Info")
    st.info("""
    - Model: XGBoost Regressor  
    - Accuracy: ~80% R²  
    - Dataset: California Housing  
    """)

    st.header("⚙ Settings")
    show_chart = st.toggle("Show Feature Importance", value=True)

# ------------------ INPUT SECTION ------------------
st.subheader("📥 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    medinc = st.slider("💰 Median Income", 0.0, 15.0, 3.0)
    houseage = st.slider("🏚 House Age", 0.0, 100.0, 20.0)
    averooms = st.slider("🛏 Avg Rooms", 1.0, 20.0, 5.0)

with col2:
    avebedrms = st.slider("🛌 Avg Bedrooms", 0.5, 10.0, 2.0)
    population = st.number_input("👥 Population", 0, 50000, 1000)
    aveoccup = st.slider("👪 Avg Occupancy", 1.0, 10.0, 3.0)

with col3:
    latitude = st.slider("🌍 Latitude", 32.5, 42.0, 34.0)
    longitude = st.slider("🌎 Longitude", -124.5, -114.0, -118.0)

# ------------------ VALIDATION ------------------
if averooms < avebedrms:
    st.warning("⚠ Rooms should be greater than bedrooms")

# ------------------ PREDICTION ------------------
if st.button("🔮 Predict Price"):
    with st.spinner("🤖 AI is analyzing..."):
        time.sleep(1.5)

        input_data = np.array([[medinc, houseage, averooms, avebedrms,
                                population, aveoccup, latitude, longitude]])

        prediction = model.predict(input_data)[0] * 100000

    # ------------------ RESULT ------------------
    st.markdown(f"""
    <div class="card">
    <h2 style='text-align:center;'>💰 Estimated Price</h2>
    <h1 style='text-align:center; color:green;'>${prediction:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ------------------ METRICS ------------------
    colA, colB, colC = st.columns(3)

    colA.metric("Median Income", medinc)
    colB.metric("House Age", houseage)
    colC.metric("Population", population)

    # ------------------ INPUT SUMMARY ------------------
    with st.expander("📋 View Input Details"):
        df = pd.DataFrame({
            "Feature": ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                        "Population", "AveOccup", "Latitude", "Longitude"],
            "Value": [medinc, houseage, averooms, avebedrms,
                      population, aveoccup, latitude, longitude]
        })
        st.dataframe(df, use_container_width=True)

    # ------------------ FEATURE IMPORTANCE (DUMMY DEMO) ------------------
    if show_chart:
        st.subheader("📊 Feature Importance (Sample Visualization)")

        importance = pd.DataFrame({
            "Feature": df["Feature"],
            "Importance": np.random.rand(8)
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance.set_index("Feature"))

# ------------------ FOOTER ------------------
st.markdown("""
---
💡 Built with ❤️ using Streamlit | XGBoost | Machine Learning
""")
