import streamlit as st
from risk_calc import calculate_diabetes_risk, calculate_lifestyle_risk

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter your health, lifestyle, and family history details.")

# User input form
with st.form("risk_form"):
    st.subheader("👤 Personal & Health Info")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    st.subheader("🏃 Lifestyle Factors")
    smoking = st.radio("Do you smoke?", ("No", "Yes"))
    alcohol = st.radio("Do you consume alcohol?", ("No", "Yes"))
    activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    sleep_hours = st.slider("Average Sleep Duration (hours)", 3, 12, 7)

    st.subheader("👪 Family History")
    family_history = st.radio("Family History of Diabetes?", ("No", "Yes"))

    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    # ML model input
    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    # Run ML prediction
    model_result = calculate_diabetes_risk(user_data)

    # Lifestyle & family risk scoring
    lifestyle_score = calculate_lifestyle_risk(smoking, alcohol, activity, sleep_hours, family_history)

    st.subheader("🔍 Risk Analysis Results")

    # Show ML model result
    if model_result == 1:
        st.error("🚨 ML Model Prediction: High Risk of Diabetes")
    else:
        st.success("✅ ML Model Prediction: Low Risk of Diabetes")

    # Show lifestyle risk score
    st.markdown(f"### 🔢 Lifestyle Risk Score: `{lifestyle_score} / 9`")

    # Risk level interpretation
    if lifestyle_score >= 6:
        st.error("⚠️ Your lifestyle risk score is high. Strongly consider preventive action.")
    elif lifestyle_score >= 3:
        st.warning("🟠 Moderate lifestyle risk. Improvements recommended.")
    else:
        st.success("🟢 Healthy lifestyle pattern.")

    st.markdown("---")
    st.subheader("🧠 Overall Health Guidance")

    if model_result == 1 or lifestyle_score >= 6:
        st.info("📌 You are in a high-risk group. Consult a doctor and take preventive steps.")
    elif model_result == 0 and lifestyle_score < 3:
        st.success("🎉 Great job! You appear to be at low risk. Maintain your healthy habits.")
    else:
        st.warning("📋 You’re at moderate risk. Improving lifestyle can reduce your future risk.")

    st.caption("⚠️ Disclaimer: This tool is for educational use. Consult a healthcare professional for medical advice.")
