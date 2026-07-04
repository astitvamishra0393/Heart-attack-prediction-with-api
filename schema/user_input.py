from pydantic import BaseModel, Field
from typing import Annotated, Literal


class UserInput(BaseModel):

    age: Annotated[int, Field(
        ..., gt=0, lt=120,
        description="Age of the individual"
    )]

    cp: Literal[0, 1, 2, 3] = Field(
        ...,
        description="Chest pain type: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic"
    )

    trestbps: float = Field(
        ...,
        description="Resting blood pressure in mmHg"
    )

    chol: float = Field(
        ...,
        description="Serum cholesterol in mg/dl"
    )

    fbs: Literal[0, 1] = Field(
        ...,
        description="Fasting blood sugar: 1 if >120 mg/dl else 0"
    )

    restecg: Literal[0, 1, 2] = Field(
        ...,
        description="ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy"
    )

    thalach: float = Field(
        ...,
        description="Maximum heart rate achieved"
    )

    exang: Literal[0, 1] = Field(
        ...,
        description="Exercise induced angina: 1=yes, 0=no"
    )

    oldpeak: float = Field(
        ...,
        description="ST depression induced by exercise"
    )

    slope: Literal[0, 1, 2] = Field(
        ...,
        description="Slope: 0=upsloping, 1=flat, 2=downsloping"
    )

    ca: Literal[0, 1, 2, 3] = Field(
        ...,
        description="Number of vessels colored (0–3)"
    )

    thal: Literal[1, 3, 6, 7] = Field(
        ...,
        description="Thal: 1/3=normal, 6=fixed defect, 7=reversible defect"
    )

    sex_male: Literal[0, 1] = Field(
        ...,
        description="Gender: 1=male, 0=female"
    )