"""Model serving API using FastAPI."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import logging
from typing import List, Dict, Any
import os
import joblib
from config import MLOpsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Classification API",
    description="MLOps pipeline model serving endpoint",
    version="1.0.0"
)

# Global model variable
model = None
scaler = None
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
class_names = ["setosa", "versicolor", "virginica"]


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str
    prediction_id: int
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    features: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 2.9, 4.3, 1.3],
                    [7.3, 2.9, 6.3, 1.8]
                ]
            }
        }


def load_model_from_mlflow(model_name: str, version: str = "latest"):
    """Load model from MLflow registry."""
    try:
        config = MLOpsConfig.from_env()
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        model_uri = f"models:/{model_name}/{version}"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(f"Model loaded from MLflow: {model_uri}")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        return None


def load_model_from_file(model_path: str):
    """Load model from local file."""
    try:
        loaded_model = joblib.load(model_path)
        logger.info(f"Model loaded from file: {model_path}")
        return loaded_model
    except Exception as e:
        logger.error(f"Failed to load model from file: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, scaler
    
    config = MLOpsConfig.from_env()
    
    # Try to load from MLflow first
    model = load_model_from_mlflow(config.model_name)
    
    # Fallback to local file
    if model is None:
        model_path = os.getenv("MODEL_PATH", "models/iris_classifier.joblib")
        if os.path.exists(model_path):
            model = load_model_from_file(model_path)
    
    if model is None:
        logger.error("Failed to load model from any source")
        raise RuntimeError("Model could not be loaded")
    
    # Load scaler if available
    scaler_path = os.getenv("SCALER_PATH", "data/scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.warning(f"Failed to load scaler: {e}")
    
    logger.info("Model serving API is ready!")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Iris Classification API", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400, 
                detail="Expected 4 features: sepal_length, sepal_width, petal_length, petal_width"
            )
        
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Apply scaling if scaler is available
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction_id = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class name and confidence
        prediction_class = class_names[prediction_id]
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
        
        return PredictionResponse(
            prediction=prediction_class,
            prediction_id=int(prediction_id),
            confidence=confidence,
            probabilities=prob_dict
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if not request.features:
            raise HTTPException(status_code=400, detail="No features provided")
        
        for i, features in enumerate(request.features):
            if len(features) != 4:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample {i}: Expected 4 features, got {len(features)}"
                )
        
        # Prepare features
        features = np.array(request.features)
        
        # Apply scaling if scaler is available
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Format results
        results = []
        for i, (pred_id, probs) in enumerate(zip(predictions, probabilities)):
            prediction_class = class_names[pred_id]
            confidence = float(max(probs))
            
            prob_dict = {
                class_names[j]: float(probs[j]) 
                for j in range(len(class_names))
            }
            
            results.append({
                "sample_id": i,
                "prediction": prediction_class,
                "prediction_id": int(pred_id),
                "confidence": confidence,
                "probabilities": prob_dict
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "feature_names": feature_names,
        "class_names": class_names,
        "n_features": len(feature_names),
        "n_classes": len(class_names)
    }
    
    # Add model parameters if available
    if hasattr(model, 'get_params'):
        info["model_parameters"] = model.get_params()
    
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)