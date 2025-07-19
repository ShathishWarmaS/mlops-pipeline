"""Kubeflow pipeline definition for iris classification."""

from kfp import dsl
from kfp.dsl import pipeline
from components import load_and_preprocess_data, train_model, evaluate_model


@pipeline(
    name="iris-classification-pipeline",
    description="MLOps pipeline for iris classification with MLflow tracking"
)
def iris_classification_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    mlflow_tracking_uri: str = "http://mlflow-service:5000",
    experiment_name: str = "iris-classification-kubeflow"
):
    """
    Kubeflow pipeline for iris classification.
    
    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the random forest
        mlflow_tracking_uri: MLflow tracking server URI
        experiment_name: MLflow experiment name
    """
    
    # Step 1: Load and preprocess data
    data_task = load_and_preprocess_data(
        test_size=test_size,
        random_state=random_state
    )
    data_task.set_display_name("Load and Preprocess Data")
    
    # Step 2: Train model
    train_task = train_model(
        dataset_input=data_task.outputs["dataset_output"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        n_estimators=n_estimators,
        random_state=random_state
    )
    train_task.set_display_name("Train Model")
    train_task.after(data_task)
    
    # Step 3: Evaluate model
    eval_task = evaluate_model(
        dataset_input=data_task.outputs["dataset_output"],
        model_input=train_task.outputs["model_output"]
    )
    eval_task.set_display_name("Evaluate Model")
    eval_task.after(train_task)
    
    # Set resource requirements (optional)
    data_task.set_cpu_limit("1")
    data_task.set_memory_limit("2Gi")
    
    train_task.set_cpu_limit("2")
    train_task.set_memory_limit("4Gi")
    
    eval_task.set_cpu_limit("1")
    eval_task.set_memory_limit("2Gi")


if __name__ == "__main__":
    # Compile pipeline for testing
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=iris_classification_pipeline,
        package_path="iris_classification_pipeline.yaml"
    )
    
    print("Pipeline compiled successfully to 'iris_classification_pipeline.yaml'")