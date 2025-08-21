import streamlit as st 
import numpy as np
import pandas as pd
import pickle

# Load model
try:
    with open("final_model (3).pkl", "rb") as f:
        model = pickle.load(f) 
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found! Please upload `final_model (3).pkl`.")
    model = None

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }
        .title {
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            color: #BB86FC;
            margin-top: 20px;
        }
        .stButton > button {
            background-color: #6200EE;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
            transition: 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #3700B3;
            transform: scale(1.05);
        }
        .result-box {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: white;
            padding: 16px;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.3);
        }
        .high-risk {
            background-color: #D32F2F;
        }
        .low-risk {
            background-color: #388E3C;
        }
        .probability {
            background-color: #FFA726;
            margin-top: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>🩺 PCOS Risk Predictor</h1>", unsafe_allow_html=True)

# Collect user inputs
with st.expander("📄 Personal Information", expanded=True):
    Age = st.slider("🎂 Age", 18, 44, 30)

with st.expander("🏥 Medical Information", expanded=True):
    BMI = st.number_input("⚖️ BMI", min_value=8, max_value=50, value=23)
    Menstrual_Irregularity = st.selectbox("🩸 Menstrual Irregularity (0 = No, 1 = Yes)", [0, 1])
    Testosterone_Level = st.number_input("🧬 Testosterone Level (ng/dL)", min_value=20, max_value=135, value=53)
    Antral_Follicle_Count = st.number_input("🧪 Antral Follicle Count", min_value=3, max_value=39, value=8)

# Predict
if st.button("🔍 Predict Risk"):
    if model is None:
        st.error("⚠️ Model is not loaded. Please ensure the model file exists.")
    else:
        try:
            input_df = pd.DataFrame([[
                Age,
                BMI,
                Menstrual_Irregularity,
                Testosterone_Level,
                Antral_Follicle_Count
            ]], columns=[
                'Age',
                'BMI',
                'Menstrual_Irregularity',
                'Testosterone_Level(ng/dL)',
                'Antral_Follicle_Count'
            ])

            prediction = model.predict(input_df)[0]
        
            if prediction == 1:
                result_text = "⚠️❌ High Risk of PCOS Detected!"
                result_class = "high-risk"
            else:
                result_text = "✅ Low Risk of PCOS — You're Good!"
                result_class = "low-risk"

            st.markdown(
                f"<div class='result-box {result_class}'>{result_text}</div>",
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")
