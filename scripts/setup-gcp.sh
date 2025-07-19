#!/bin/bash

# Setup script for deploying MLOps pipeline on Google Cloud Platform

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required environment variables are set
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo -e "${RED}Error: GOOGLE_CLOUD_PROJECT environment variable is not set${NC}"
    exit 1
fi

if [ -z "$GOOGLE_CLOUD_REGION" ]; then
    echo -e "${YELLOW}Warning: GOOGLE_CLOUD_REGION not set, defaulting to us-central1${NC}"
    export GOOGLE_CLOUD_REGION="us-central1"
fi

if [ -z "$CLUSTER_NAME" ]; then
    echo -e "${YELLOW}Warning: CLUSTER_NAME not set, defaulting to mlops-cluster${NC}"
    export CLUSTER_NAME="mlops-cluster"
fi

echo -e "${GREEN}Setting up MLOps pipeline on Google Cloud Platform${NC}"
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Region: $GOOGLE_CLOUD_REGION"
echo "Cluster: $CLUSTER_NAME"

# Enable required APIs
echo -e "${GREEN}Enabling required Google Cloud APIs...${NC}"
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable ml.googleapis.com
gcloud services enable storage.googleapis.com

# Create GKE cluster
echo -e "${GREEN}Creating GKE cluster...${NC}"
gcloud container clusters create $CLUSTER_NAME \
    --region=$GOOGLE_CLOUD_REGION \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-autorepair \
    --enable-autoupgrade \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5 \
    --disk-size=50GB \
    --disk-type=pd-standard \
    --enable-ip-alias \
    --network=default \
    --subnetwork=default

# Get cluster credentials
echo -e "${GREEN}Getting cluster credentials...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --region=$GOOGLE_CLOUD_REGION

# Create service account for workload identity
echo -e "${GREEN}Creating service account...${NC}"
gcloud iam service-accounts create mlops-pipeline-sa \
    --display-name="MLOps Pipeline Service Account" \
    --description="Service account for MLOps pipeline operations"

# Grant necessary permissions
echo -e "${GREEN}Granting permissions to service account...${NC}"
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:mlops-pipeline-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:mlops-pipeline-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/ml.admin"

gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:mlops-pipeline-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"

# Create and download service account key
echo -e "${GREEN}Creating service account key...${NC}"
gcloud iam service-accounts keys create gcp-service-account-key.json \
    --iam-account=mlops-pipeline-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com

# Create Kubernetes secret for service account
echo -e "${GREEN}Creating Kubernetes secret...${NC}"
kubectl create namespace mlops-pipeline --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic gcp-service-account-key \
    --from-file=key.json=gcp-service-account-key.json \
    --namespace=mlops-pipeline

# Install Kubeflow Pipelines
echo -e "${GREEN}Installing Kubeflow Pipelines...${NC}"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.5"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=1.8.5"

# Wait for Kubeflow Pipelines to be ready
echo -e "${GREEN}Waiting for Kubeflow Pipelines to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=ml-pipeline --timeout=300s -n kubeflow

# Create GCS bucket for artifacts
echo -e "${GREEN}Creating GCS bucket for MLflow artifacts...${NC}"
gsutil mb -p $GOOGLE_CLOUD_PROJECT -c STANDARD -l $GOOGLE_CLOUD_REGION gs://${GOOGLE_CLOUD_PROJECT}-mlflow-artifacts || echo "Bucket might already exist"

echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Build and push Docker images: ./docker/build.sh"
echo "2. Push images to GCR:"
echo "   docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-mlflow:latest"
echo "   docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-training:latest"
echo "3. Update image names in k8s manifests with your project ID"
echo "4. Deploy the pipeline: ./scripts/deploy.sh"
echo ""
echo "Kubeflow Pipelines UI will be available at:"
echo "kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80"
echo "Then open http://localhost:8080"