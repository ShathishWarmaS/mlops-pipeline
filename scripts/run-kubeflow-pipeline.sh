#!/bin/bash

# Script to compile and run Kubeflow pipeline

set -e

echo "Compiling and running Kubeflow pipeline..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Compile pipeline
cd kubeflow
python pipeline.py

echo "Pipeline compiled to iris_classification_pipeline.yaml"

# Check if Kubeflow endpoint is available
if [ -z "$KUBEFLOW_ENDPOINT" ]; then
    echo "Warning: KUBEFLOW_ENDPOINT not set. Using port-forward to access Kubeflow."
    echo "Run: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
    echo "Then set: export KUBEFLOW_ENDPOINT=http://localhost:8080"
    echo "And re-run this script to submit the pipeline."
else
    # Submit pipeline
    python run_pipeline.py --kubeflow-endpoint $KUBEFLOW_ENDPOINT
    echo "Pipeline submitted successfully!"
fi

cd ..