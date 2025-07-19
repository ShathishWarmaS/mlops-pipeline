"""Configuration settings for MLOps pipeline."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLOpsConfig:
    """Configuration class for MLOps pipeline."""
    
    # MLflow settings
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "iris-classification"
    mlflow_artifact_root: Optional[str] = None
    
    # Model settings
    model_name: str = "iris-classifier"
    model_version: str = "1.0.0"
    test_size: float = 0.2
    random_state: int = 42
    
    # Data settings
    data_path: str = "data/iris.csv"
    
    # GCS settings (for GCP deployment)
    gcs_bucket: Optional[str] = os.getenv("GCS_BUCKET")
    project_id: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT")
    region: str = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    
    # Kubeflow settings
    kubeflow_namespace: str = "kubeflow-user-example-com"
    pipeline_name: str = "iris-classification-pipeline"
    
    @classmethod
    def from_env(cls) -> "MLOpsConfig":
        """Create config from environment variables."""
        return cls(
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            region=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
        )