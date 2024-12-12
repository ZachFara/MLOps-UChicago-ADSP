from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from typing import Dict, Any
import yaml
import logging
from src.utils.logging_utils import setup_logging

# Initialize logging
logger = setup_logging(__name__)

# Load config
def load_config() -> Dict[str, Any]:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

app = FastAPI(title="Social Media Time-Waster Predictor")

class PredictionInput(BaseModel):
    features: Dict[str, Any]

class PredictionOutput(BaseModel):
    prediction: float
    prediction_probability: float

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Load the model
        model = mlflow.pyfunc.load_model("mlruns/0/latest/artifacts/model")
        
        # Convert input to DataFrame
        features_df = pd.DataFrame([input_data.features])
        
        # Make prediction
        prediction = model.predict(features_df)
        
        return PredictionOutput(
            prediction=float(prediction[0]),
            prediction_probability=float(prediction[0])
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    try:
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_latest_versions("social_media_classifier", stages=["Production"])[0]
        return {
            "version": model_info.version,
            "creation_timestamp": model_info.creation_timestamp,
            "current_stage": model_info.current_stage
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 