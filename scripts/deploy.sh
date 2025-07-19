#!/bin/bash

# Deploy the MLOps pipeline to GKE

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying MLOps pipeline to Kubernetes...${NC}"

# Apply Kubernetes manifests
echo -e "${GREEN}Creating namespace...${NC}"
kubectl apply -f k8s/namespace.yaml

echo -e "${GREEN}Applying ConfigMap...${NC}"
kubectl apply -f k8s/configmap.yaml

echo -e "${GREEN}Deploying MLflow server...${NC}"
kubectl apply -f k8s/mlflow-deployment.yaml

# Wait for MLflow to be ready
echo -e "${GREEN}Waiting for MLflow server to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=mlflow-server --timeout=300s -n mlops-pipeline

echo -e "${GREEN}Starting training job...${NC}"
kubectl apply -f k8s/training-job.yaml

# Monitor job status
echo -e "${GREEN}Monitoring training job...${NC}"
kubectl wait --for=condition=complete job/iris-training-job --timeout=600s -n mlops-pipeline

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo ""
echo "Useful commands:"
echo "- Check MLflow server: kubectl get pods -n mlops-pipeline -l app=mlflow-server"
echo "- Check training job: kubectl get jobs -n mlops-pipeline"
echo "- View training logs: kubectl logs -n mlops-pipeline job/iris-training-job"
echo "- Port forward to MLflow UI: kubectl port-forward -n mlops-pipeline svc/mlflow-service 5000:5000"
echo "- Port forward to Kubeflow UI: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
echo ""
echo "Access MLflow UI at: http://localhost:5000"
echo "Access Kubeflow Pipelines UI at: http://localhost:8080"