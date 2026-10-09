
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="NeuroWell | Developer Wellness AI",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        MODEL_DIR / "neurowell_ann.keras"
    )
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "features.json", "r") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_assets()

st.title("🧠 NeuroWell")
st.subheader("ANN-Based Developer Wellness Risk Intelligence")
st.write(
    "Explore how workload and lifestyle indicators map to a "
    "simulated wellness-risk category."
)

st.warning(
    "Educational demo using synthetic data and simulated labels. "
    "This is not a medical tool and cannot diagnose burnout."
)

st.divider()

st.sidebar.header("Developer Indicators")

with st.sidebar.form("wellness_form"):
    working_hours = st.slider("Working hours per day", 4.0, 14.0, 9.0, 0.5)
    sleep_hours = st.slider("Sleep hours per night", 3.5, 9.0, 7.0, 0.5)
    coding_hours = st.slider("Coding hours per day", 1.0, 10.0, 6.0, 0.5)
    meetings = st.slider("Meetings per day", 0, 8, 3)
    breaks = st.slider("Breaks per day", 0, 7, 3)
    pressure = st.slider("Deadline pressure (1–10)", 1, 10, 5)
    satisfaction = st.slider("Job satisfaction (1–10)", 1, 10, 7)
    balance = st.slider("Work-life balance (1–10)", 1, 10, 7)
    consecutive_days = st.slider("Consecutive working days", 1, 14, 5)
    tasks = st.slider("Tasks completed per day", 1, 15, 7)

    submitted = st.form_submit_button(
        "Analyze Indicators",
        use_container_width=True
    )

if submitted:
    values = {
        "working_hours": working_hours,
        "sleep_hours": sleep_hours,
        "coding_hours": coding_hours,
        "meetings_per_day": meetings,
        "breaks_per_day": breaks,
        "deadline_pressure": pressure,
        "job_satisfaction": satisfaction,
        "work_life_balance": balance,
        "consecutive_working_days": consecutive_days,
        "tasks_completed": tasks
    }

    input_df = pd.DataFrame([values], columns=features)
    scaled_input = scaler.transform(input_df)

    probabilities = model.predict(scaled_input, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))

    labels = ["Low", "Medium", "High"]
    predicted_label = labels[predicted_class]

    st.subheader("Analysis Result")

    col1, col2 = st.columns(2)
    col1.metric("Predicted simulated category", predicted_label)
    col2.metric(
        "Model probability for this category",
        f"{probabilities[predicted_class] * 100:.1f}%"
    )

    st.caption(
        "This percentage is the model's estimated probability for its "
        "predicted class, not a validated real-world burnout risk score."
    )

    st.write("Model probabilities by category")
    probability_df = pd.DataFrame({
        "Category": labels,
        "Model probability (%)": probabilities * 100
    })
    st.bar_chart(
        probability_df.set_index("Category")
    )

    st.subheader("Your entered indicators")
    st.dataframe(input_df, use_container_width=True)

    st.info(
        "Use this demonstration to discuss ANN classification, "
        "feature scaling, softmax probabilities, and model deployment. "
        "For actual wellbeing concerns, seek appropriate human support."
    )
else:
    st.info("Adjust the indicators in the sidebar and click Analyze Indicators.")
