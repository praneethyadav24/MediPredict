import numpy as np
import joblib

# Load trained model
model = joblib.load("models/diabetes_model.pkl")

def calculate_lifestyle_risk(smoking, alcohol, activity, sleep_hours, family_history):
    """
    Calculate a lifestyle-based risk score based on user habits.
    """
    score = 0
    if smoking == "Yes":
        score += 2
    if alcohol == "Yes":
        score += 1
    if activity == "Low":
        score += 2
    elif activity == "Moderate":
        score += 1
    if sleep_hours < 6:
        score += 1
    if family_history == "Yes":
        score += 3
    return score

def calculate_diabetes_risk(user_data):
    """
    Predict diabetes risk using trained ML model.
    
    user_data = {
        "Pregnancies": int,
        "Glucose": int,
        "BloodPressure": int,
        "SkinThickness": int,
        "Insulin": int,
        "BMI": float,
        "DiabetesPedigreeFunction": float,
        "Age": int
    }
    """
    input_array = np.array([[ 
        user_data["Pregnancies"],
        user_data["Glucose"],
        user_data["BloodPressure"],
        user_data["SkinThickness"],
        user_data["Insulin"],
        user_data["BMI"],
        user_data["DiabetesPedigreeFunction"],
        user_data["Age"]
    ]])

    prediction = model.predict(input_array)[0]
    return prediction  # 0 = No diabetes, 1 = Likely diabetes

# Test block
if __name__ == "__main__":
    test_input = {
        "Pregnancies": 2,
        "Glucose": 130,
        "BloodPressure": 80,
        "SkinThickness": 25,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.4,
        "Age": 35
    }

    result = calculate_diabetes_risk(test_input)
    print("Prediction (1 = diabetic, 0 = not):", result)
