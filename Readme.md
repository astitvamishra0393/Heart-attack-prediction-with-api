# Project Structure
Heart_attack_predictor_shap/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── model/
│   ├── predict.py
│   ├── heart_attack_model.pkl
│   └── shap_explainer.pkl
│
├── schema/
│   └── user_input.py
│
└── venv/

## Project Overview

### Description

--A production-ready machine learning system that predicts heart attack risk with 91% recall and 0.88 ROC-AUC. Implements stratified             
cross-validation, hyperparameter tuning with Optuna, SHAP explainability, and deploys via FastAPI + Docker for clinical-grade interpretability.

## Project Workflow

Raw Data
    ↓
Data Cleaning
    ↓
Missing Value Handling
    ↓
Outlier Detection
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Train-Test Split
    ↓
Stratified Cross Validation
    ↓
Model Selection
    ↓
Logistic Regression
Random Forest
AdaBoost
XGBoost
    ↓
Performance Comparison
    ↓
Selecting top 2 models
    ↓
Optuna Hyperparameter Optimization
    ↓
Best Model Selection
    ↓
SHAP Explainability
    ↓
Model Saving and Explainer Saving using joblib
    ↓
FastAPI Deployment
    ↓
Docker Containerization
    

## Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early prediction can help healthcare professionals identify high-risk patients and take preventive action.

The objective of this project is to develop a machine learning system capable of predicting heart disease risk using patient clinical attributes while providing explainable predictions.

### Source
UCI Heart Disease Dataset

- Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### Dataset Information

| Attribute | Description |
|------------|-------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest Pain Type |
| trestbps | Resting Blood Pressure |
| chol | Serum Cholesterol |
| fbs | Fasting Blood Sugar |
| restecg | Resting ECG Results |
| thalach | Maximum Heart Rate Achieved |
| exang | Exercise Induced Angina |
| oldpeak | ST Depression |
| slope | Slope of Peak Exercise ST Segment |
| ca | Number of Major Vessels |
| thal | Thalassemia Status |
| target | Heart Disease (Yes/No) |

### Dataset Size

- Rows: 303
- Features: 13

# Exploratory Data Analysis

 

Performed
- Class Imbalance using value_count(bar plot)
- Outlier Detection using bar graph and box plot of numerical columns
- Correlation Heatmap
- Potential Outliers Detection
- Feature vs Target Analysis

Visualizations were generated using:

- Matplotlib
- Seaborn

# Feature Engineering

Implemented:

- Categorical Encoding
- Feature Transformation
- Feature Selection
- Data Validation

## Cross validation

Performed using 
- cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
 )


# Models Evaluated

The following algorithms were trained and compared:

### Logistic Regression

- Interpretable baseline model

### Random Forest Classifier

- Ensemble-based model
- Selected as final model after tuning

### AdaBoost Classifier

- Boosting-based ensemble method

### XGBoost Classifier

- Gradient Boosting algorithm

## Results of baseline models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 81.37% | 80.32% | 87.04% | 83.52% | 0.8876 |
| **AdaBoost** | 81.40% | 82.99% | 83.30% | 83.04% | 0.8762 |
| **Random Forest** | 80.95% | 81.43% | 84.79% | 83.01% | 0.8879 |
| **XGBoost** | 80.18% | 81.43% | 83.30% | 82.23% | 0.8661 |

### Key Observations

- **Best Recall:** Logistic Regression (87.04%) - catches most cases
- **Best Precision:** AdaBoost (82.99%) - fewer false alarms
- **Best ROC-AUC:** Random Forest (0.8879) - best discrimination
- **Selection Criteria:** Logistic Regression selected for **highest recall** (medical priority: catch cases)

### Why Recall?

In medical applications, **missing a heart attack case (false negative) is costlier than over-caution (false positive).**
- Recall = 87% means catching 87 out of 100 actual cases
- Precision = 80% means 20% of flagged patients are false alarms (acceptable for follow-up testing)

--- Scaled data for logistic regression but it did not made much of a difference so droped the whole scaling idea

# Hyperparameter Optimization

Optuna was used to optimize the top-performing models.
Models optimized:
    -- Logistic Regression 
    -- Random Forest

Optimized Parameters(random forest):

- n_estimators
- max_depth
- min_samples_split
- Additional algorithm-specific parameters

Optimized Parameters(logistic regression):

- C


# Final Model Performance

### Random Forest (Optuna Tuned)

| Metric | Score |
|----------|----------|
| Accuracy | 0.82 |
| Precision | 0.80 |
| Recall | 0.97 |
| F1 Score | 0.88 |
| ROC-AUC | 0.91 |

### Confusion Matrix

[[20  8]
 [ 1 32]]

 # Explainable AI (SHAP)

SHAP was integrated to provide transparent and explainable predictions.

### Global Explainability

- SHAP Summary Plot
- SHAP Feature Importance Plot

### Human-Readable Explanations

| Feature | Explanation |
|----------|-------------|
| cp | Chest pain type strongly influenced risk |
| ca | Number of affected vessels increased risk |
| exang | Exercise-induced angina contributed to prediction |

# Model Serialization

Model artifacts were saved using Joblib.

```python
joblib.dump(model, "heart_attack_model.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
```

Saved Artifacts:

- Final Model
- SHAP Explainer
- Feature Metadata

# API Endpoints

## Base URL

http://localhost:8000


## Home Endpoint

Returns a welcome message indicating that the API is running.

### Request

http GET /

json
{
  "message": "Heart Attack Prediction Interface"
}


## Health Check Endpoint

Verifies that the API is running and the model has been loaded successfully.

### Request

http
GET /health


### Response

json
{
  "status": "OK",
  "model_loaded": true
}


## Predict Heart Attack Risk

Predicts whether a patient is at risk of heart disease based on the provided clinical features.

### Request

http
POST /predict

### Sample Request Body

json
{
  "age": 45,
  "cp": 1,
  "trestbps": 130,
  "chol": 233,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 1,
  "ca": 0,
  "thal": 3,
  "sex_male": 1
}

### Sample Response

json
{
  "prediction": 1,
  "probability": [0.23, 0.77]
}

### Response Fields

| Field       | Description                                   |
| ----------- | --------------------------------------------- |
| prediction  | Predicted class (0 = Low Risk, 1 = High Risk) |
| probability | Class probabilities returned by the model     |

---

## Explain Prediction

Provides SHAP-based feature importance for the given input.

### Request

http
POST /explain

### Sample Request Body

json
{
  "age": 45,
  "cp": 1,
  "trestbps": 130,
  "chol": 233,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 1,
  "ca": 0,
  "thal": 3,
  "sex_male": 1
}

### Sample Response

json
{
  "top_3_influence": [
    {
      "feature": "chol",
      "shap_value": 0.42
    },
    {
      "feature": "age",
      "shap_value": 0.31
    },
    {
      "feature": "thalach",
      "shap_value": -0.28
    }
  ]
}

### Response Fields

| Field      | Description                            |
| ---------- | -------------------------------------- |
| feature    | Feature contributing to the prediction |
| shap_value | SHAP contribution value                |


## Predict with Explanation

Returns prediction, probability, top contributing features, and a human-readable explanation.

### Request

http
POST /predict-with-explanation


### Sample Request Body

json
{
  "age": 45,
  "cp": 1,
  "trestbps": 130,
  "chol": 233,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 1,
  "ca": 0,
  "thal": 3,
  "sex_male": 1
}

### Sample Response

json
{
  "prediction": 1,
  "probability": [0.23, 0.77],
  "top_3_features": [
    {
      "feature": "chol",
      "shap_value": 0.42
    },
    {
      "feature": "age",
      "shap_value": 0.31
    },
    {
      "feature": "thalach",
      "shap_value": -0.28
    }
  ],
  "explanation": {
    "result": "HIGH RISK",
    "reasoning": [
      "chol increases risk significantly",
      "age increases risk significantly",
      "thalach reduces risk significantly"
    ]
  }
}

# Run FastAPI
uvicorn main:app --reload

## Dockerization

FROM python:3.12-slim

WORKDIR /main

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

### PULL

- docker pull astitva0393/heart-attack-api:latest

### Run

- Run: docker run -p 8000:8000 astitva0393/heart-attack-api

# Requirements

annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.14.1
click==8.4.2
cloudpickle==3.1.2
colorama==0.4.6
fastapi==0.139.0
h11==0.16.0
idna==3.18
joblib==1.5.3
llvmlite==0.48.0
narwhals==2.23.0
numba==0.66.0
numpy==2.4.6
packaging==26.2
pandas==3.0.3
pydantic==2.13.4
pydantic_core==2.46.4
python-dateutil==2.9.0.post0
scikit-learn==1.9.0
scipy==1.18.0
shap==0.52.0
six==1.17.0
slicer==0.0.8
starlette==1.3.1
threadpoolctl==3.6.0
tqdm==4.68.3
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.2
uvicorn==0.50.0
