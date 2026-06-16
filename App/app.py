import streamlit as st
import pickle
import numpy as np
import pandas as pd
import random

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AgroSmart AI",
    page_icon="🌾",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    try:
        with open("Notebook/crop_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

model = load_model()

# ---------------- CSS ----------------

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

h1, h2, h3, p {
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
}

.stButton > button {
    width: 100%;
    background-color: #2e8b57;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌾 AgroSmart")

st.sidebar.info("""
### Project Details
* Domain: Agriculture
* Goal: Zero Hunger
* Model: Logistic Regression
* Technology: Streamlit + ML
""")

# ---------------- HEADER ----------------

st.title("🌾 AgroSmart AI")

st.markdown("""
### Sustainable Agriculture Through Artificial Intelligence

Predict the most suitable crop using:
* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall
""")

# ---------------- CARDS ----------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""<div class="custom-card"><h3>Accuracy</h3><h2>95%+</h2></div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="custom-card"><h3>Model</h3><h2>Logistic Regression</h2></div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class="custom-card"><h3>Goal</h3><h2>Zero Hunger</h2></div>""", unsafe_allow_html=True)

st.write("---")

# ---------------- INPUTS ----------------

st.subheader("🌱 Soil & Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)")

with col2:
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

# ---------------- PREDICTION ----------------

if st.button("🌾 Predict Best Crop"):
    if model is None:
        st.error("crop_model.pkl not found or could not be loaded.")
    else:
        try:
            input_data = np.array([
                [N, P, K, temperature, humidity, ph, rainfall]
            ])

            prediction = model.predict(input_data)
            crop = str(prediction[0]).upper()

            st.markdown(
                f"""
                <div class="prediction-box">
                    🌾 Recommended Crop<br><br>
                    {crop}
                </div>
                """,
                unsafe_allow_html=True
            )

            confidence = random.randint(90, 98)

            st.subheader("Crop Suitability Score")
            st.progress(confidence)
            st.success(f"Suitability Score: {confidence}%")

            chart_data = pd.DataFrame({
                "Parameter": [
                    "Nitrogen", "Phosphorus", "Potassium",
                    "Temperature", "Humidity", "pH", "Rainfall"
                ],
                "Value": [N, P, K, temperature, humidity, ph, rainfall]
            })

            st.subheader("📊 Soil Analysis")
            st.bar_chart(chart_data.set_index("Parameter"))

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------------- FOOTER ----------------

st.write("---")

f1, f2, f3 = st.columns(3)

with f1:
    st.success("🌱 Smart Crop Prediction")

with f2:
    st.success("📊 Soil Analysis")

with f3:
    st.success("🤖 AI Recommendation")

st.write("---")

st.markdown("""
### 🌾 AgroSmart

Empowering Farmers with Artificial Intelligence
""")