#!/bin/bash

# Build Docker images for the MLOps pipeline

set -e

echo "Building MLOps pipeline Docker images..."

# Build MLflow server image
echo "Building MLflow server image..."
docker build -f docker/Dockerfile.mlflow -t iris-mlflow:latest .

# Build training image
echo "Building training image..."
docker build -f docker/Dockerfile.training -t iris-training:latest .

# Build serving image
echo "Building serving image..."
docker build -f docker/Dockerfile.serving -t iris-serving:latest .

echo "Docker images built successfully!"

# Tag images for GCR (Google Container Registry) if PROJECT_ID is set
if [ ! -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "Tagging images for Google Container Registry..."
    
    docker tag iris-mlflow:latest gcr.io/$GOOGLE_CLOUD_PROJECT/iris-mlflow:latest
    docker tag iris-training:latest gcr.io/$GOOGLE_CLOUD_PROJECT/iris-training:latest
    docker tag iris-serving:latest gcr.io/$GOOGLE_CLOUD_PROJECT/iris-serving:latest
    
    echo "Images tagged for GCR. Push with:"
    echo "  docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-mlflow:latest"
    echo "  docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-training:latest"
    echo "  docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-serving:latest"
else
    echo "Set GOOGLE_CLOUD_PROJECT environment variable to tag images for GCR"
fi

echo "Build complete!"