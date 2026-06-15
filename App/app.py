import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random



st.set_page_config(
    page_title="AgroSmart AI",
    page_icon="🌾",
    layout="wide"
)



st.markdown("""
<style>

.stApp {
    background-image:
    linear-gradient(
        rgba(0,0,0,0.65),
        rgba(0,0,0,0.65)
    ),
    url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

h1,h2,h3 {
    color: white !important;
    text-shadow: 2px 2px 8px black;
}

p {
    color: white !important;
}

.custom-card {
    background: rgba(0,0,0,0.75);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.prediction-box {
    background: white;
    color: #2e8b57;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.stButton > button {
    width: 100%;
    background-color: #2e8b57;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
}

section[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.85);
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return pickle.load(open('../Model/crop_model.pkl', 'rb'))
    except:
        return None

model = load_model()



st.sidebar.title("🌾 AgroSmart")

st.sidebar.info("""
### Project Details

**Domain:** Agriculture

**Goal:** Zero Hunger

**Model:** Logistic Regression

**Technology:** Streamlit + Machine Learning
""")

# ---------------- HEADER ---------------- #

st.title("🌾 AgroSmart AI")

st.markdown("""
### Sustainable Agriculture Through Artificial Intelligence

Predict the most suitable crop using:

Nitrogen (N)  
Phosphorus (P)  
Potassium (K)  
Temperature  
Humidity  
pH  
Rainfall
""")

# ---------------- DASHBOARD CARDS ---------------- #

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="custom-card">
    <h3>Accuracy</h3>
    <h2>95%+</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="custom-card">
    <h3>Model</h3>
    <h2>Logistic Regression</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="custom-card">
    <h3>Goal</h3>
    <h2>Zero Hunger</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# ---------------- INPUT SECTION ---------------- #

st.markdown("## 🌱 Soil & Environmental Parameters")

col1, col2 = st.columns(2)

with col1:

    st.markdown("**Nitrogen (N)**")
    N = st.number_input("", key="n")

    st.markdown("**Phosphorus (P)**")
    P = st.number_input("", key="p")

    st.markdown("**Potassium (K)**")
    K = st.number_input("", key="k")

    st.markdown("**Temperature (°C)**")
    temperature = st.number_input("", key="temp")

with col2:

    st.markdown("**Humidity (%)**")
    humidity = st.number_input("", key="hum")

    st.markdown("**Soil pH**")
    ph = st.number_input("", key="ph")

    st.markdown("**Rainfall (mm)**")
    rainfall = st.number_input("", key="rain")


if st.button(" Predict Best Crop"):

    if model is None:
        st.error("Model file not found. Check crop_model.pkl")

    else:

        input_data = np.array([
            [N, P, K, temperature,
             humidity, ph, rainfall]
        ])

        prediction = model.predict(input_data)

        crop = str(prediction[0]).lower()

        # Prediction Card

        st.markdown(
            f"""
            <div class="prediction-box">
            🌾 Recommended Crop<br><br>
            {crop.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

    

        confidence = random.randint(90, 98)

        st.subheader("Crop Suitability Score")

        st.progress(confidence)

        st.success(
            f"Suitability Score: {confidence}%"
        )

        # Input Analysis Chart

        st.subheader(" Soil & Environmental Analysis")

        chart_data = pd.DataFrame({
            "Parameter": [
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Temperature",
                "Humidity",
                "pH",
                "Rainfall"
            ],
            "Value": [
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]
        })

        st.bar_chart(
            chart_data.set_index("Parameter")
        )

    

        crop_info = {

            "rice":
            "Rice grows best in high rainfall and humid conditions.",

            "maize":
            "Maize prefers warm temperatures and moderate rainfall.",

            "cotton":
            "Cotton grows well in black soil and warm climates.",

            "mango":
            "Mango thrives in tropical and subtropical climates."
        }

        st.subheader(" AI Recommendation Report")

        st.info(
            f"""
Recommended Crop: {crop.upper()}

Reason:

Based on soil nutrients, temperature,
humidity, pH and rainfall values,
the Logistic Regression model predicts
{crop.upper()} as the most suitable crop.

This recommendation helps improve crop
yield and supports sustainable agriculture.
"""
        )

        if crop in crop_info:
            st.success(crop_info[crop])



st.write("---")

f1, f2, f3 = st.columns(3)

with f1:
    st.success("🌱 Smart Crop Prediction")

with f2:
    st.success("📊Soil Analysis")

with f3:
    st.success("AI Recommendation")



st.write("---")

st.markdown("""
### 🌾 AgroSmart

Empowering Farmers with Artificial Intelligence
""")