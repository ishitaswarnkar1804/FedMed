from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.predict import predict


app = FastAPI(
    title="FedMed API",
    description="Medical prediction API",
    version="1.0.0"
)


class PatientData(BaseModel):

    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float


FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


@app.get("/")
def home():

    return {
        "message": "FedMed API is running"
    }


@app.post("/predict")
def make_prediction(patient: PatientData):

    values = patient.model_dump()

    input_data = [
        values["mean_radius"],
        values["mean_texture"],
        values["mean_perimeter"],
        values["mean_area"],
        values["mean_smoothness"],
        values["mean_compactness"],
        values["mean_concavity"],
        values["mean_concave_points"],
        values["mean_symmetry"],
        values["mean_fractal_dimension"],

        values["radius_error"],
        values["texture_error"],
        values["perimeter_error"],
        values["area_error"],
        values["smoothness_error"],
        values["compactness_error"],
        values["concavity_error"],
        values["concave_points_error"],
        values["symmetry_error"],
        values["fractal_dimension_error"],

        values["worst_radius"],
        values["worst_texture"],
        values["worst_perimeter"],
        values["worst_area"],
        values["worst_smoothness"],
        values["worst_compactness"],
        values["worst_concavity"],
        values["worst_concave_points"],
        values["worst_symmetry"],
        values["worst_fractal_dimension"]
    ]

    result = predict(input_data)

    return result
