from fastapi import FastAPI
from schema.user_input import UserInput
from model.predict import (
    predict_output,
    explain_output,
    predict_with_explanation
)

# Import model safely for health check
from model.predict import model

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Heart Attack Prediction Interface 🚀"}


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None
    }


# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: UserInput):
    return predict_output(data.model_dump())


# ---------------- EXPLAIN ----------------
@app.post("/explain")
def explain(data: UserInput):
    return explain_output(data.model_dump())


# ---------------- COMBINED ----------------
@app.post("/predict-with-explanation")
def predict_full(data: UserInput):
    return predict_with_explanation(data.model_dump())
from schema.user_input import UserInput
from model.predict import (
    predict_output,
    explain_output,
    predict_with_explanation
)

# Import model safely for health check
from model.predict import model

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Heart Attack Prediction Interface"}


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None
    }


# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: UserInput):
    return predict_output(data.model_dump())


# ---------------- EXPLAIN ----------------
@app.post("/explain")
def explain(data: UserInput):
    return explain_output(data.model_dump())


# ---------------- COMBINED ----------------
@app.post("/predict-with-explanation")
def predict_full(data: UserInput):
    return predict_with_explanation(data.model_dump())