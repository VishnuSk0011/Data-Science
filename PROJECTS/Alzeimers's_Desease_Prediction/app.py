import streamlit as st
import pandas as pd
from pickle import load

# -----------------------------
# Load model & features
# -----------------------------
lgbm = load(open("lgbm.pkl", "rb"))
features = load(open("features.pkl", "rb"))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Alzheimer's Prediction")
st.title("🧠 Alzheimer's Disease Prediction System")
st.write("Enter key clinical and lifestyle details:")

# -----------------------------
# User Inputs (IMPORTANT FEATURES in DATASET ORDER)
# -----------------------------

age = st.number_input("Age", 1, 120, 60)

diet = st.number_input("Diet Quality", 0.0, 10.0, 5.0, step=0.1)

chol_total = st.number_input("Total Cholesterol", 100.0, 400.0, 200.0, step=0.1)
chol_ldl = st.number_input("LDL Cholesterol", 50.0, 300.0, 120.0, step=0.1)
chol_trig = st.number_input("Triglycerides", 50.0, 500.0, 150.0, step=0.1)

mmse = st.number_input("MMSE Score", 0.0, 30.0, 25.0, step=0.1)
functional = st.number_input("Functional Assessment", 0.0, 10.0, 5.0, step=0.1)

memory = st.selectbox("Memory Complaints", ["No", "Yes"])
memory_encoded = 1 if memory == "Yes" else 0

behavior = st.number_input("Behavioral Problems", 0, 10, 2)

adl = st.number_input("ADL (Daily Activity Level)", 0.0, 10.0, 5.0, step=0.1)


# -----------------------------
# Create FULL feature vector (32)
# -----------------------------
input_dict = dict.fromkeys(features, 0)

# Fill important features
input_dict["Age"] = age
input_dict["DietQuality"] = diet
input_dict["CholesterolTotal"] = chol_total
input_dict["CholesterolLDL"] = chol_ldl
input_dict["CholesterolTriglycerides"] = chol_trig
input_dict["MMSE"] = mmse
input_dict["FunctionalAssessment"] = functional
input_dict["MemoryComplaints"] = memory_encoded
input_dict["BehavioralProblems"] = behavior
input_dict["ADL"] = adl

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = lgbm.predict(input_df)
    probability = lgbm.predict_proba(input_df)[0][prediction[0]]

    if prediction[0] == 1:
        st.error("⚠️ High likelihood of Alzheimer's Disease")
    else:
        st.success("✅ Low likelihood of Alzheimer's Disease")

    st.write(f"Prediction Confidence: **{probability:.2f}**")

