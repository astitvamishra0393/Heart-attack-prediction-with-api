import joblib
from pathlib import Path
import pandas as pd
import numpy as np

FEATURES = [
    "age", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "sex_male"
]

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "heart_attack_model.pkl"
shap_path = BASE_DIR / "shap_explainer.pkl"

# ---------------- LOAD ----------------
model = joblib.load(model_path)
explainer = joblib.load(shap_path)


# ---------------- CORE SHAP FUNCTION ----------------
def get_shap_values(df):

    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    values = np.array(values).flatten()
    values = [float(v) for v in values]

    return dict(zip(FEATURES, values))


# ---------------- PREDICT ONLY ----------------
def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])
    df = df[FEATURES]

    prediction = model.predict(df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": probability
    }


# ---------------- EXPLANATION ----------------
def explain_output(user_input: dict):

    df = pd.DataFrame([user_input])
    df = df[FEATURES]

    shap_dict = get_shap_values(df)

    sorted_features = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    top_3 = [
        {"feature": f, "shap_value": round(v, 4)}
        for f, v in sorted_features
    ]

    return {
        "shap_values": shap_dict,
        "top_3_influence": top_3
    }


# ---------------- HUMAN EXPLANATION ----------------
def generate_explanation(prediction, top_3):

    risk = "HIGH RISK" if prediction == 1 else "LOW RISK"

    reasoning = []
    for item in top_3:
        feat = item["feature"]
        val = item["shap_value"]

        direction = "increases risk" if val > 0 else "reduces risk"
        reasoning.append(f"{feat} {direction} significantly")

    return {
        "result": risk,
        "reasoning": reasoning
    }


# ---------------- FINAL API FUNCTION ----------------
def predict_with_explanation(user_input: dict):

    df = pd.DataFrame([user_input])
    df = df[FEATURES]

    prediction = model.predict(df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[0].tolist()

    shap_dict = get_shap_values(df)

    sorted_features = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    top_3 = [
        {"feature": f, "shap_value": round(v, 4)}
        for f, v in sorted_features
    ]

    explanation_text = generate_explanation(prediction, top_3)

    return {
        "prediction": int(prediction),
        "probability": probability,
        "top_3_features": top_3,
        "explanation": explanation_text
    }