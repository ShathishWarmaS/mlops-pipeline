"""Script to run the Kubeflow pipeline."""

import kfp
from kfp.dsl import pipeline
from pipeline import iris_classification_pipeline
import argparse
import os


def run_pipeline(
    kubeflow_endpoint: str,
    experiment_name: str = "iris-classification-experiment",
    run_name: str = "iris-classification-run",
    mlflow_uri: str = "http://mlflow-service:5000"
):
    """Run the iris classification pipeline on Kubeflow."""
    
    # Create client
    client = kfp.Client(host=kubeflow_endpoint)
    
    # Get or create experiment
    try:
        experiment = client.get_experiment(experiment_name=experiment_name)
        experiment_id = experiment.experiment_id
    except:
        experiment = client.create_experiment(name=experiment_name)
        experiment_id = experiment.experiment_id
    
    # Pipeline arguments
    arguments = {
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "mlflow_tracking_uri": mlflow_uri,
        "experiment_name": "iris-classification-kubeflow"
    }
    
    # Submit pipeline run
    run_result = client.run_pipeline(
        experiment_id=experiment_id,
        job_name=run_name,
        pipeline_func=iris_classification_pipeline,
        arguments=arguments
    )
    
    print(f"Pipeline run submitted: {run_result.run_id}")
    print(f"Run URL: {kubeflow_endpoint}/#/runs/details/{run_result.run_id}")
    
    return run_result


def main():
    parser = argparse.ArgumentParser(description="Run Kubeflow pipeline")
    parser.add_argument("--kubeflow-endpoint", required=True, 
                       help="Kubeflow Pipelines endpoint URL")
    parser.add_argument("--experiment-name", default="iris-classification-experiment",
                       help="Experiment name")
    parser.add_argument("--run-name", default="iris-classification-run",
                       help="Pipeline run name")
    parser.add_argument("--mlflow-uri", default="http://mlflow-service:5000",
                       help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    run_result = run_pipeline(
        kubeflow_endpoint=args.kubeflow_endpoint,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        mlflow_uri=args.mlflow_uri
    )
    
    print("Pipeline run completed successfully!")


if __name__ == "__main__":
    main()