"""ML model training and evaluation with MLflow tracking."""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from config import MLOpsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisClassifier:
    """Iris classification model with MLflow tracking."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=config.random_state
        )
        self.is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train the model with MLflow tracking."""
        
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        
        with mlflow.start_run():
            logger.info("Starting model training...")
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", self.model.n_estimators)
            mlflow.log_param("random_state", self.model.random_state)
            mlflow.log_param("test_size", self.config.test_size)
            
            # Train the model
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("cv_mean_accuracy", cv_mean)
            mlflow.log_metric("cv_std_accuracy", cv_std)
            
            # Log the model
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name=self.config.model_name
            )
            
            # Create and log confusion matrix
            self._log_confusion_matrix(y_test, y_pred_test)
            
            # Log classification report
            class_report = classification_report(y_test, y_pred_test, output_dict=True)
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)
            
            # Log feature importance
            self._log_feature_importance()
            
            logger.info(f"Model training completed. Test accuracy: {test_accuracy:.4f}")
            
            return {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
                "run_id": mlflow.active_run().info.run_id
            }
    
    def _log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Create and log confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    yticklabels=['Setosa', 'Versicolor', 'Virginica'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
    
    def _log_feature_importance(self):
        """Log feature importance plot."""
        if not self.is_fitted:
            return
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        importances = self.model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        
        plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        # Log individual feature importances as metrics
        for name, importance in zip(feature_names, importances):
            mlflow.log_metric(f"feature_importance_{name}", importance)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")