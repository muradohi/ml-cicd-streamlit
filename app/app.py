import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
import yaml

st.set_page_config(page_title="Employee Churn Predictor", layout="wide")
st.title("🏢 Employee Churn Predictor")

with open("/Users/murad/ml-cicd-streamlit/ml-cicd-streamlit/configs/config.yml") as f:
    cfg = yaml.safe_load(f)

model_path = cfg["model"]["path"]

if not model_path:
    st.error("⚠️ Model not found. Please train it first.")
    st.stop()

model = load(model_path)
st.success("✅ Model loaded successfully!")

st.sidebar.header("Employee Information")

# Numeric inputs
numeric_inputs = {col: st.sidebar.number_input(col, value=0.0) for col in cfg["numeric_features"]}

# Categorical inputs
categorical_inputs = {col: st.sidebar.text_input(col, value="") for col in cfg["categorical_features"]}

# Combine into DataFrame
input_df = pd.DataFrame([{**numeric_inputs, **categorical_inputs}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"🔴 Employee likely to leave (prob={prob:.2f})")
    else:
        st.success(f"🟢 Employee likely to stay (prob={1-prob:.2f})")

st.caption("Demo: Logistic Regression + GridSearchCV + Cross-Validation + Streamlit UI")
