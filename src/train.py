"""Training script for the iris classification model."""

import click
import logging
from pathlib import Path
from data_loader import DataLoader
from model import IrisClassifier
from config import MLOpsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--config-file', type=str, help='Path to config file (optional)')
@click.option('--mlflow-uri', type=str, default='http://localhost:5000', 
              help='MLflow tracking URI')
@click.option('--experiment-name', type=str, default='iris-classification',
              help='MLflow experiment name')
def main(config_file: str, mlflow_uri: str, experiment_name: str):
    """Train the iris classification model."""
    
    # Load configuration
    if config_file:
        # In a real implementation, you might load from YAML/JSON
        config = MLOpsConfig.from_env()
    else:
        config = MLOpsConfig(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_experiment_name=experiment_name
        )
    
    logger.info("Starting training pipeline...")
    logger.info(f"MLflow URI: {config.mlflow_tracking_uri}")
    logger.info(f"Experiment: {config.mlflow_experiment_name}")
    
    try:
        # Load and prepare data
        data_loader = DataLoader(
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        X_train, X_test, y_train, y_test = data_loader.load_iris_data()
        
        # Save data for later use
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        data_loader.save_data(X_train, X_test, y_train, y_test, str(data_dir))
        
        # Initialize and train model
        classifier = IrisClassifier(config)
        results = classifier.train(X_train, y_train, X_test, y_test)
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "iris_classifier.joblib"
        classifier.save_model(str(model_path))
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()