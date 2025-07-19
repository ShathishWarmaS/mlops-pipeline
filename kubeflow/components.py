"""Kubeflow pipeline components for the iris classification pipeline."""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


@component(
    base_image="python:3.9",
    packages_to_install=[
        "scikit-learn==1.3.2", 
        "pandas==2.1.4", 
        "numpy==1.24.3",
        "mlflow==2.8.1"
    ]
)
def load_and_preprocess_data(
    dataset_output: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42
) -> NamedTuple('DataStats', [('train_samples', int), ('test_samples', int), ('features', int)]):
    """Load and preprocess the iris dataset."""
    
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    import os
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    os.makedirs(dataset_output.path, exist_ok=True)
    
    np.save(f"{dataset_output.path}/X_train.npy", X_train_scaled)
    np.save(f"{dataset_output.path}/X_test.npy", X_test_scaled)
    np.save(f"{dataset_output.path}/y_train.npy", y_train)
    np.save(f"{dataset_output.path}/y_test.npy", y_test)
    
    # Save scaler
    with open(f"{dataset_output.path}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names and target names
    with open(f"{dataset_output.path}/feature_names.pkl", 'wb') as f:
        pickle.dump(iris.feature_names, f)
    
    with open(f"{dataset_output.path}/target_names.pkl", 'wb') as f:
        pickle.dump(iris.target_names, f)
    
    # Return statistics
    from collections import namedtuple
    DataStats = namedtuple('DataStats', ['train_samples', 'test_samples', 'features'])
    return DataStats(
        train_samples=len(X_train_scaled),
        test_samples=len(X_test_scaled),
        features=X_train_scaled.shape[1]
    )


@component(
    base_image="python:3.9",
    packages_to_install=[
        "scikit-learn==1.3.2", 
        "numpy==1.24.3",
        "mlflow==2.8.1",
        "joblib==1.3.2"
    ]
)
def train_model(
    dataset_input: Input[Dataset],
    model_output: Output[Model],
    mlflow_tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "iris-classification",
    n_estimators: int = 100,
    random_state: int = 42
) -> NamedTuple('TrainingMetrics', [('train_accuracy', float), ('test_accuracy', float)]):
    """Train the iris classification model."""
    
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    
    # Load processed data
    X_train = np.load(f"{dataset_input.path}/X_train.npy")
    X_test = np.load(f"{dataset_input.path}/X_test.npy")
    y_train = np.load(f"{dataset_input.path}/y_train.npy")
    y_test = np.load(f"{dataset_input.path}/y_test.npy")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        os.makedirs(model_output.path, exist_ok=True)
        model_path = f"{model_output.path}/model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_estimators": n_estimators,
            "random_state": random_state
        }
        
        import json
        with open(f"{model_output.path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    # Return metrics
    from collections import namedtuple
    TrainingMetrics = namedtuple('TrainingMetrics', ['train_accuracy', 'test_accuracy'])
    return TrainingMetrics(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy
    )


@component(
    base_image="python:3.9",
    packages_to_install=[
        "scikit-learn==1.3.2", 
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "joblib==1.3.2"
    ]
)
def evaluate_model(
    dataset_input: Input[Dataset],
    model_input: Input[Model],
    metrics_output: Output[Metrics]
) -> NamedTuple('EvaluationResults', [('accuracy', float), ('precision', float), ('recall', float), ('f1_score', float)]):
    """Evaluate the trained model and generate detailed metrics."""
    
    import numpy as np
    import joblib
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    
    # Load test data
    X_test = np.load(f"{dataset_input.path}/X_test.npy")
    y_test = np.load(f"{dataset_input.path}/y_test.npy")
    
    # Load target names
    with open(f"{dataset_input.path}/target_names.pkl", 'rb') as f:
        target_names = pickle.load(f)
    
    # Load model
    model = joblib.load(f"{model_input.path}/model.joblib")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{metrics_output.path}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    # Save detailed metrics
    detailed_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": class_report,
        "confusion_matrix": cm.tolist()
    }
    
    with open(f"{metrics_output.path}/evaluation_metrics.json", 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    # Return summary metrics
    from collections import namedtuple
    EvaluationResults = namedtuple('EvaluationResults', ['accuracy', 'precision', 'recall', 'f1_score'])
    return EvaluationResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1
    )