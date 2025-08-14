from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.utils.logging_config import setup_logging

# --- Constants and Setup ---
MODEL_NAME = "IrisClassifier"
MODEL_ALIAS = "production"

# Initialize logging
logger = setup_logging()

# Global variable to hold the model
model = None


# --- Lifespan Manager for Model Loading ---
# This is the modern way to handle startup/shutdown events in FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager to handle application lifespan events.
    Loads the ML model on startup and cleans up on shutdown.
    """
    global model
    logger.info("Application startup: Loading ML model...")
    try:
        # Use the portable relative path for the tracking URI
        mlflow.set_tracking_uri("./mlruns")

        # Load model using the alias (e.g., "production")
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(
            f"Successfully loaded model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'"
        )
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        model = None

    yield  # The application runs while the lifespan context is active

    logger.info("Application shutdown: Cleaning up resources.")
    # You can add cleanup code here if needed


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Iris Model API",
    description="API for serving the Iris classification model.",
    version="0.1.0",
    lifespan=lifespan,  # Use the lifespan manager
)

# Instrument with Prometheus metrics
Instrumentator().instrument(app).expose(app)


# --- API Data Models (Pydantic Schemas) ---
class IrisInput(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

    class Config:
        # Updated from 'schema_extra' to 'json_schema_extra' to fix warning
        json_schema_extra = {
            "example": {
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            }
        }


class PredictionOut(BaseModel):
    prediction: int
    class_name: str


# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check API status."""
    return {"status": "Iris model API is running."}


@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict(request: Request, data: IrisInput):
    """
    Accepts Iris flower features and returns the predicted species.
    - 0: Setosa
    - 1: Versicolor
    - 2: Virginica
    """
    client_host = request.client.host
    logger.info(f"Received prediction request from {client_host}: {data.dict()}")

    # This is the new, robust error handling
    if model is None:
        logger.error("Model is not loaded. Returning 503 Service Unavailable.")
        raise HTTPException(
            status_code=503, detail="Model is not available. Please try again later."
        )

    try:
        # Column names must match the model's training columns exactly
        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        input_data = pd.DataFrame(
            [
                [
                    data.sepal_length_cm,
                    data.sepal_width_cm,
                    data.petal_length_cm,
                    data.petal_width_cm,
                ]
            ],
            columns=feature_names,
        )

        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])

        class_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        predicted_class_name = class_map.get(predicted_class, "Unknown")

        logger.info(f"Prediction result: {predicted_class} ({predicted_class_name})")
        return PredictionOut(
            prediction=predicted_class, class_name=predicted_class_name
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to make a prediction.")
