"""Client for making inference requests to the model serving API."""

import requests
import json
import numpy as np
from typing import List, Dict, Any
import click
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for making inference requests."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def predict_single(self, features: List[float]) -> Dict[str, Any]:
        """Make a single prediction."""
        try:
            payload = {"features": features}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Single prediction failed: {e}")
            return {"error": str(e)}
    
    def predict_batch(self, features: List[List[float]]) -> Dict[str, Any]:
        """Make batch predictions."""
        try:
            payload = {"features": features}
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction failed: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}


def generate_sample_data(n_samples: int = 5) -> List[List[float]]:
    """Generate sample iris data for testing."""
    np.random.seed(42)
    
    # Generate realistic iris features
    samples = []
    for _ in range(n_samples):
        # Sepal length: 4.0-8.0
        sepal_length = np.random.uniform(4.0, 8.0)
        # Sepal width: 2.0-4.5
        sepal_width = np.random.uniform(2.0, 4.5)
        # Petal length: 1.0-7.0
        petal_length = np.random.uniform(1.0, 7.0)
        # Petal width: 0.1-2.5
        petal_width = np.random.uniform(0.1, 2.5)
        
        samples.append([sepal_length, sepal_width, petal_length, petal_width])
    
    return samples


@click.group()
def cli():
    """Inference client for iris classification API."""
    pass


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API base URL')
def health(url: str):
    """Check API health."""
    client = InferenceClient(url)
    result = client.health_check()
    print(json.dumps(result, indent=2))


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API base URL')
def info(url: str):
    """Get model information."""
    client = InferenceClient(url)
    result = client.get_model_info()
    print(json.dumps(result, indent=2))


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API base URL')
@click.option('--sepal-length', type=float, required=True, help='Sepal length')
@click.option('--sepal-width', type=float, required=True, help='Sepal width')
@click.option('--petal-length', type=float, required=True, help='Petal length')
@click.option('--petal-width', type=float, required=True, help='Petal width')
def predict(url: str, sepal_length: float, sepal_width: float, 
           petal_length: float, petal_width: float):
    """Make a single prediction."""
    client = InferenceClient(url)
    features = [sepal_length, sepal_width, petal_length, petal_width]
    result = client.predict_single(features)
    print(json.dumps(result, indent=2))


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API base URL')
@click.option('--samples', default=5, help='Number of sample predictions')
def test(url: str, samples: int):
    """Test the API with sample data."""
    client = InferenceClient(url)
    
    print("=== Health Check ===")
    health_result = client.health_check()
    print(json.dumps(health_result, indent=2))
    
    if health_result.get("status") != "healthy":
        print("API is not healthy, skipping tests")
        return
    
    print("\n=== Model Info ===")
    info_result = client.get_model_info()
    print(json.dumps(info_result, indent=2))
    
    print(f"\n=== Single Prediction ===")
    # Classic setosa sample
    setosa_features = [5.1, 3.5, 1.4, 0.2]
    single_result = client.predict_single(setosa_features)
    print(f"Input: {setosa_features}")
    print(json.dumps(single_result, indent=2))
    
    print(f"\n=== Batch Predictions ({samples} samples) ===")
    batch_features = generate_sample_data(samples)
    batch_result = client.predict_batch(batch_features)
    
    if "predictions" in batch_result:
        for i, (features, prediction) in enumerate(zip(batch_features, batch_result["predictions"])):
            print(f"\nSample {i+1}: {features}")
            print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
    else:
        print(json.dumps(batch_result, indent=2))


@cli.command()
@click.option('--url', default='http://localhost:8000', help='API base URL')
@click.option('--file', type=click.Path(exists=True), required=True, help='JSON file with features')
def predict_file(url: str, file: str):
    """Make predictions from a JSON file."""
    client = InferenceClient(url)
    
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        
        if "features" in data:
            # Single prediction
            if isinstance(data["features"][0], (int, float)):
                result = client.predict_single(data["features"])
            # Batch prediction
            else:
                result = client.predict_batch(data["features"])
        else:
            print("Invalid file format. Expected {'features': [...]} or {'features': [[...], [...]]}")
            return
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    cli()