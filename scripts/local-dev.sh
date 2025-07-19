#!/bin/bash

# Local development setup script

set -e

echo "Setting up local development environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data models logs

echo "Local environment setup complete!"
echo ""
echo "To activate the environment: source venv/bin/activate"
echo "To run local training: python src/train.py"
echo "To start local MLflow server: mlflow server --host 0.0.0.0 --port 5000"
echo "To run with Docker Compose: docker-compose -f docker/docker-compose.yml up"