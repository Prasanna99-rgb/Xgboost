import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page config
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("XGBR.pkl", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'XGBR.pkl' is in the app directory.")
        st.stop()

model = load_model()

# Title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("""
This app estimates the **median house value** for California districts based on 
1990 census data. Simply adjust the sliders below and click **Predict**.
""")

# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    - **Model:** XGBoost Regressor  
    - **R² Score:** ~0.80 (on test set)  
    - **Features:** 8 attributes from the California Housing dataset  
    - **Target:** Median house value in **USD**  
    """)
    st.caption("Data source: scikit-learn's California housing dataset")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    medinc = st.slider(
        "💰 Median Income", 
        min_value=0.0, max_value=15.0, value=3.0, step=0.1,
        help="Median income in block group (in $10,000s)"
    )
    houseage = st.slider(
        "🏚️ House Age", 
        min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="Median house age in block group (years)"
    )
    averooms = st.slider(
        "🛏️ Avg Rooms per Household", 
        min_value=1.0, max_value=20.0, value=5.0, step=0.1,
        help="Average number of rooms per dwelling"
    )
    avebedrms = st.slider(
        "🛌 Avg Bedrooms per Household", 
        min_value=0.5, max_value=10.0, value=2.0, step=0.1,
        help="Average number of bedrooms per dwelling"
    )

with col2:
    population = st.number_input(
        "👥 Population", 
        min_value=0, max_value=50000, value=1000, step=100,
        help="Total population in block group"
    )
    aveoccup = st.slider(
        "👪 Avg Occupancy", 
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="Average number of household members"
    )
    latitude = st.slider(
        "🌍 Latitude", 
        min_value=32.5, max_value=42.0, value=34.0, step=0.01,
        help="Latitude of block group"
    )
    longitude = st.slider(
        "🌎 Longitude", 
        min_value=-124.5, max_value=-114.0, value=-118.0, step=0.01,
        help="Longitude of block group"
    )

# Predict button
if st.button("🔮 Predict House Price", type="primary"):
    # Prepare input array
    input_features = np.array([[medinc, houseage, averooms, avebedrms,
                                population, aveoccup, latitude, longitude]])
    
    # Predict (target is in $100,000 units)
    prediction = model.predict(input_features)[0] * 100_000
    
    # Display result
    st.success(f"### Estimated Price: **${prediction:,.2f}**")
    
    # Optional: Show input summary
    with st.expander("📋 Your input summary"):
        input_df = pd.DataFrame({
            "Feature": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                        "Population", "AveOccup", "Latitude", "Longitude"],
            "Value": [medinc, houseage, averooms, avebedrms, 
                      population, aveoccup, latitude, longitude]
        })
        st.dataframe(input_df, use_container_width=True)
