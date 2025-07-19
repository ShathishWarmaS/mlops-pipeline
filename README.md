# Enterprise MLOps Pipeline with Advanced AI Operations

A complete **enterprise-grade MLOps platform** featuring comprehensive model lifecycle management, advanced monitoring, automated retraining, and AI governance - deployable on Google Cloud Platform with full observability.

## 🚀 **Enterprise-Grade Features**

### **Core MLOps Platform**
🏗️ **Complete Dev/Staging/Prod Lifecycle** - Multi-environment model management with automatic experiment setup  
📊 **70+ Metrics Per Run** - Comprehensive tracking including feature importance, cross-validation, and confidence analysis  
🎨 **8 Rich Visualizations** - ROC curves, confusion matrices, feature importance plots, and more  
🔄 **Model Promotion Workflows** - CLI tools for promoting models between environments  
📋 **Enterprise Reporting** - Detailed JSON reports with performance comparisons  

### **🔬 Advanced Monitoring & Observability**
📈 **Real-time Model Performance Monitoring** - Prometheus metrics, Grafana dashboards, custom alerting  
🕵️ **Data Drift Detection** - Statistical drift analysis with PSI, KS tests, and Wasserstein distance  
📊 **Distributed Tracing** - OpenTelemetry integration with Jaeger for request tracing  
🎯 **A/B Testing Framework** - Statistical significance testing for model comparisons  
⚡ **System Health Monitoring** - CPU, memory, disk usage with automatic alerting  

### **🤖 Intelligent Automation**
🔄 **Automated Retraining Pipeline** - Trigger-based retraining with drift, performance, and schedule triggers  
🧠 **Model Explainability** - SHAP, LIME, and permutation importance for model interpretation  
🔒 **Security & Governance** - Model lineage tracking, audit trails, compliance reporting  
⚙️ **Smart Resource Management** - Auto-scaling, load balancing, cost optimization  

### **🛠️ DevOps & CI/CD**
🚀 **Advanced GitHub Actions Pipeline** - Automated testing, security scanning, deployment  
🐳 **Multi-Environment Docker Configs** - Development, staging, production containers  
☁️ **Cloud-Native Architecture** - GKE deployment with Istio service mesh  
📦 **Model Registry & Artifacts** - Versioned model storage with metadata tracking  

### **🎯 Production-Ready Excellence**
🔐 **Enterprise Security** - RBAC, secret management, vulnerability scanning  
📊 **Comprehensive Logging** - Structured logging with correlation IDs and context  
🎪 **Multi-Model Serving** - Canary deployments, traffic routing, fallback strategies  
📈 **Business Metrics Integration** - KPI tracking, cost analysis, ROI measurement  

Perfect for **Fortune 500 companies** requiring enterprise-grade AI operations with complete governance and observability!

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Prep     │───▶│   Training      │───▶│   Evaluation    │
│   Component     │    │   Component     │    │   Component     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │   MLflow Tracking   │
                    │     Server          │
                    └─────────────────────┘
```

## 📁 **Enterprise Project Structure**

```
mlops-pipeline/
├── 🏗️ src/                          # Core MLOps Platform
│   ├── config.py                   # Configuration management
│   ├── data_loader.py              # Data loading utilities
│   ├── model.py                    # Enhanced ML model with MLflow tracking
│   ├── train.py                    # Multi-environment training script
│   ├── mlflow_manager.py           # Enhanced MLflow lifecycle management
│   ├── model_lifecycle.py          # Model promotion and lifecycle CLI
│   ├── serve.py                    # FastAPI model serving
│   ├── inference_client.py         # API testing client
│   │
│   ├── 🔬 monitoring.py            # Advanced monitoring & alerting system
│   ├── 🕵️ drift_detector.py        # Data drift detection & analysis
│   ├── 🎯 ab_testing.py            # A/B testing framework
│   ├── 🤖 automated_retraining.py  # Intelligent retraining pipeline
│   ├── 📊 observability.py         # Comprehensive logging & tracing
│   └── 🧠 model_explainability.py  # SHAP, LIME, interpretability tools
│
├── 🐳 docker/                       # Multi-Environment Containers
│   ├── Dockerfile.training         # Training environment
│   ├── Dockerfile.mlflow           # MLflow server
│   ├── Dockerfile.serving          # Model serving
│   ├── docker-compose.yml          # Basic orchestration
│   ├── docker-compose.mlflow.yml   # Advanced MLflow lifecycle
│   └── build.sh                    # Container build scripts
│
├── ☁️ k8s/                          # Cloud-Native Kubernetes
│   ├── namespace.yaml              # Environment isolation
│   ├── mlflow-deployment.yaml      # MLflow tracking server
│   ├── training-job.yaml           # Training workloads
│   ├── serving-deployment.yaml     # Model serving
│   ├── monitoring/                 # Prometheus, Grafana configs
│   └── configmap.yaml              # Configuration management
│
├── 🚀 .github/workflows/            # Advanced CI/CD Pipeline
│   └── mlops-ci-cd.yml             # Complete automation workflow
│
├── 🔄 kubeflow/                     # Pipeline Orchestration
│   ├── components.py               # Pipeline components
│   ├── pipeline.py                 # Pipeline definition
│   └── run_pipeline.py             # Pipeline execution script
│
├── 🛠️ scripts/                      # Deployment & Operations
│   ├── setup-gcp.sh                # GCP infrastructure setup
│   ├── deploy.sh                   # Kubernetes deployment
│   ├── local-dev.sh                # Local development setup
│   ├── run-kubeflow-pipeline.sh    # Pipeline execution
│   └── monitoring-setup.sh         # Observability stack setup
│
├── 📊 configs/                      # Enterprise Configuration
│   ├── monitoring_config.json      # Monitoring & alerting
│   ├── drift_config.json           # Drift detection settings
│   ├── retraining_config.json      # Automated retraining
│   ├── ab_test_config.json         # A/B testing parameters
│   └── observability_config.json   # Logging & tracing
│
├── 🧪 tests/                        # Comprehensive Testing
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── security/                   # Security tests
│   └── performance/                # Performance tests
│
├── 📋 docs/                         # Enterprise Documentation
│   ├── architecture.md             # System architecture
│   ├── deployment-guide.md         # Deployment procedures
│   ├── monitoring-guide.md         # Observability setup
│   ├── security-guide.md           # Security best practices
│   └── troubleshooting.md          # Issue resolution
│
├── 📈 dashboards/                   # Monitoring Dashboards
│   ├── grafana/                    # Grafana dashboard configs
│   ├── prometheus/                 # Prometheus rules & alerts
│   └── custom/                     # Custom visualization tools
│
├── 🔐 security/                     # Security & Compliance
│   ├── rbac/                       # Role-based access control
│   ├── policies/                   # Security policies
│   └── audit/                      # Audit trail configs
│
├── 📦 artifacts/                    # Model & Data Artifacts
│   ├── models/                     # Trained model storage
│   ├── data/                       # Dataset storage
│   ├── reports/                    # Analysis reports
│   └── visualizations/             # Generated plots & charts
│
├── ⚙️ requirements.txt              # Python dependencies (60+ packages)
├── 🐳 docker-compose.override.yml   # Local development overrides
├── 🔧 pyproject.toml               # Python project configuration
├── 📝 CHANGELOG.md                 # Version history & updates
└── 📖 README.md                    # This comprehensive guide
```

## 🎯 **Enterprise Feature Deep Dive**

### 🔬 **Advanced Monitoring & Observability**

Our enterprise monitoring system provides comprehensive observability across the entire MLOps lifecycle:

#### **Real-time Performance Monitoring**
```bash
# Start comprehensive monitoring system
python src/monitoring.py

# Monitor specific model performance
python -c "
from src.monitoring import MLOpsMonitor
monitor = MLOpsMonitor()
status = monitor.get_monitoring_dashboard_data()
print(f'System health: {status}')
"

# Set up Prometheus metrics endpoint
curl http://localhost:8000/metrics
```

#### **Data Drift Detection**
```bash
# Configure drift detection
python src/drift_detector.py

# Run drift analysis
python -c "
from src.drift_detector import DriftDetectionPipeline
pipeline = DriftDetectionPipeline()
# Automatic statistical analysis with PSI, KS tests, Wasserstein distance
report = pipeline.detect_and_analyze(current_data)
print(f'Drift detected: {report.drift_detected}')
print(f'Affected features: {report.affected_features}')
"
```

#### **Distributed Tracing with OpenTelemetry**
```bash
# Enable distributed tracing
export JAEGER_ENDPOINT=http://localhost:14268/api/traces
python src/observability.py

# View traces in Jaeger UI
open http://localhost:16686
```

### 🎯 **A/B Testing Framework**

Statistical A/B testing for model performance comparison:

```bash
# Setup A/B test experiment
python -c "
from src.ab_testing import ExperimentConfig, ABTestExperiment

config = ExperimentConfig(
    name='model_v2_test',
    traffic_split=0.2,  # 20% traffic to new model
    duration_days=7,
    success_metrics=['accuracy', 'latency'],
    significance_level=0.05
)

experiment = ABTestExperiment(config)
experiment.setup_models(model_a, model_b)
"

# Monitor A/B test results
python src/ab_testing.py status --experiment model_v2_test
```

### 🤖 **Automated Retraining Pipeline**

Intelligent retraining with multiple trigger types:

```bash
# Start automated retraining orchestrator
python src/automated_retraining.py

# Configure retraining triggers
cat > retraining_config.json << EOF
{
  "triggers": [
    {
      "name": "drift_trigger",
      "trigger_type": "drift",
      "threshold": 0.1,
      "enabled": true,
      "priority": 1
    },
    {
      "name": "performance_trigger", 
      "trigger_type": "performance",
      "threshold": 0.05,
      "enabled": true,
      "priority": 2
    },
    {
      "name": "scheduled_trigger",
      "trigger_type": "schedule",
      "enabled": true,
      "conditions": {
        "frequency": "weekly",
        "day_of_week": "sunday"
      }
    }
  ]
}
EOF
```

### 🧠 **Model Explainability & Interpretability**

Comprehensive model interpretation with SHAP, LIME, and custom techniques:

```bash
# Generate model explanations
python -c "
from src.model_explainability import ModelExplainabilityPipeline

pipeline = ModelExplainabilityPipeline(model, X_train, X_test, y_test)

# Local explanations for specific predictions
local_explanations = pipeline.explain_predictions(X_sample)

# Global model behavior analysis
global_explanations = pipeline.generate_global_explanations()

# Comprehensive interpretability report
report = pipeline.generate_model_report()
print(f'Report saved: {report}')
"

# View SHAP visualizations
ls explainability_results/
# Output: global_explanation_shap.png, feature_importance_plots/, etc.
```

### 📊 **Advanced CI/CD Pipeline**

Enterprise-grade GitHub Actions workflow with:

- **Automated Testing**: Unit, integration, security, performance tests
- **Model Validation**: Drift detection, performance benchmarks
- **Security Scanning**: Vulnerability analysis, dependency checking
- **Multi-Environment Deployment**: Staging → Production with approval gates
- **Monitoring Integration**: Health checks, smoke tests, alerting

```yaml
# Triggered on push to main branch
name: MLOps CI/CD Pipeline
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        type: choice
        options: [staging, production]

jobs:
  quality-checks:     # Code quality & security
  tests:             # Unit & integration tests  
  model-training:    # Multi-environment training
  docker-build:      # Container building & scanning
  deploy-staging:    # Automated staging deployment
  ab-testing:        # A/B test setup
  deploy-production: # Manual production deployment
  monitoring:        # Observability stack setup
```

## 🚀 **Quick Start Guide**

### Prerequisites

- Python 3.9+
- Docker
- kubectl
- gcloud CLI
- Google Cloud Project with billing enabled

### 1. Local Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd mlops-pipeline

# Setup local environment
./scripts/local-dev.sh

# Activate virtual environment
source venv/bin/activate

# Run enhanced training with environment support
python src/train.py --environment development --run-name "local_dev"

# Start comprehensive MLflow UI
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5001

# Check model status across environments
python src/model_lifecycle.py status --mlflow-uri ./mlruns
```

### 2. Docker Development

#### Basic Docker Compose (Simple MLflow + Training)

```bash
# Start MLflow server
docker-compose -f docker/docker-compose.yml up mlflow

# Run development training
docker-compose -f docker/docker-compose.yml --profile dev up

# Run staging training  
docker-compose -f docker/docker-compose.yml --profile staging up

# Run production training
docker-compose -f docker/docker-compose.yml --profile prod up

# Run serving API
docker-compose -f docker/docker-compose.yml --profile serving up

# Complete development pipeline
docker-compose -f docker/docker-compose.yml --profile dev --profile serving up
```

#### Enhanced MLflow Docker Compose (Full Lifecycle Management)

```bash
# Start MLflow with health checks
docker-compose -f docker/docker-compose.mlflow.yml up mlflow

# Development environment training
docker-compose -f docker/docker-compose.mlflow.yml --profile dev up

# Staging environment training
docker-compose -f docker/docker-compose.mlflow.yml --profile staging up

# Production environment training
docker-compose -f docker/docker-compose.mlflow.yml --profile prod up

# All training environments
docker-compose -f docker/docker-compose.mlflow.yml --profile training up

# Model serving with health checks
docker-compose -f docker/docker-compose.mlflow.yml --profile serving up

# Model lifecycle management
docker-compose -f docker/docker-compose.mlflow.yml --profile lifecycle up

# Model promotion workflow
FROM_ENV=staging TO_ENV=production docker-compose -f docker/docker-compose.mlflow.yml --profile promotion up

# Inference testing
docker-compose -f docker/docker-compose.mlflow.yml --profile test up

# Complete MLOps pipeline (all profiles)
docker-compose -f docker/docker-compose.mlflow.yml \
  --profile dev --profile staging --profile prod \
  --profile serving --profile lifecycle up
```

#### Environment-Specific Training

```bash
# Development with custom run name
TIMESTAMP=$(date +%s) docker-compose -f docker/docker-compose.mlflow.yml --profile dev up

# Production deployment with timestamp
TIMESTAMP=$(date +%s) docker-compose -f docker/docker-compose.mlflow.yml --profile prod up

# Custom model promotion
FROM_ENV=development TO_ENV=staging docker-compose -f docker/docker-compose.mlflow.yml --profile promotion up
```

## ☁️ Google Cloud Deployment

### Step 1: Environment Setup

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"
export CLUSTER_NAME="mlops-cluster"

# Copy environment template
cp .env.example .env
# Edit .env with your values
```

### Step 2: GCP Infrastructure Setup

```bash
# Run GCP setup script (creates GKE cluster, service accounts, etc.)
./scripts/setup-gcp.sh
```

This script will:
- Enable required GCP APIs
- Create GKE cluster
- Set up service accounts and IAM roles
- Install Kubeflow Pipelines
- Create GCS bucket for artifacts

### Step 3: Build and Push Images

```bash
# Build Docker images
./docker/build.sh

# Push to Google Container Registry
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-mlflow:latest
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-training:latest
docker push gcr.io/$GOOGLE_CLOUD_PROJECT/iris-serving:latest
```

### Step 4: Update Kubernetes Manifests

Replace `YOUR_PROJECT_ID` in the following files with your actual project ID:
- `k8s/mlflow-deployment.yaml`
- `k8s/training-job.yaml`
- `k8s/serving-deployment.yaml`

```bash
# Quick replacement
sed -i "s/YOUR_PROJECT_ID/$GOOGLE_CLOUD_PROJECT/g" k8s/*.yaml
```

### Step 5: Deploy to Kubernetes

```bash
# Deploy the pipeline
./scripts/deploy.sh
```

### Step 6: Access the Services

```bash
# Port forward to MLflow UI
kubectl port-forward -n mlops-pipeline svc/mlflow-service 5000:5000

# Port forward to Kubeflow Pipelines UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

- MLflow UI: http://localhost:5000
- Kubeflow Pipelines UI: http://localhost:8080
- Model Serving API: http://localhost:8000 (after deploying serving service)

## 🔄 Running Kubeflow Pipelines

### Option 1: Using the Web UI

1. Access Kubeflow Pipelines UI at http://localhost:8080
2. Upload the compiled pipeline: `kubeflow/iris_classification_pipeline.yaml`
3. Create an experiment and run the pipeline

### Option 2: Using the Python Script

```bash
# Set Kubeflow endpoint
export KUBEFLOW_ENDPOINT=http://localhost:8080

# Compile and run pipeline
./scripts/run-kubeflow-pipeline.sh
```

### Option 3: Manual Pipeline Compilation and Execution

```bash
cd kubeflow

# Compile pipeline
python pipeline.py

# Submit pipeline run
python run_pipeline.py --kubeflow-endpoint http://localhost:8080
```

## 🚀 Model Serving & Inference

### Deploy Model Serving API

```bash
# Deploy serving service to Kubernetes
kubectl apply -f k8s/serving-deployment.yaml

# Port forward to access the API
kubectl port-forward -n mlops-pipeline svc/iris-serving-service 8000:8000
```

### API Endpoints

The serving API provides the following endpoints:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /docs` - OpenAPI documentation (Swagger UI)

### Using the Inference Client

```bash
# Install client dependencies
pip install click requests

# Health check
python src/inference_client.py health --url http://localhost:8000

# Get model info
python src/inference_client.py info --url http://localhost:8000

# Make a single prediction
python src/inference_client.py predict \
  --url http://localhost:8000 \
  --sepal-length 5.1 \
  --sepal-width 3.5 \
  --petal-length 1.4 \
  --petal-width 0.2

# Test with sample data
python src/inference_client.py test --url http://localhost:8000 --samples 5

# Predict from JSON file
python src/inference_client.py predict-file \
  --url http://localhost:8000 \
  --file examples/sample_requests.json
```

### API Testing

```bash
# Run comprehensive API tests
./scripts/test-serving.sh

# Test with custom URL
API_URL=http://your-api-url.com ./scripts/test-serving.sh
```

### Example API Usage

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 2.9, 4.3, 1.3],
      [7.3, 2.9, 6.3, 1.8]
    ]
  }'

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

## 🎯 Enhanced MLflow Model Lifecycle

### Multi-Environment Training

Train models in different environments with comprehensive tracking:

```bash
# Development environment
python src/train.py --environment development --run-name "dev_v1" --mlflow-uri ./mlruns

# Staging environment  
python src/train.py --environment staging --run-name "staging_validation" --mlflow-uri ./mlruns

# Production environment
python src/train.py --environment production --run-name "prod_deploy" --mlflow-uri ./mlruns
```

### Model Lifecycle Management

Use the comprehensive CLI for model lifecycle operations:

```bash
# Check model status across all environments
python src/model_lifecycle.py status --mlflow-uri ./mlruns

# Compare performance across environments
python src/model_lifecycle.py compare --mlflow-uri ./mlruns --output-file comparison.json

# Generate comprehensive model report
python src/model_lifecycle.py report --mlflow-uri ./mlruns --output-file report.json

# Promote model from staging to production
python src/model_lifecycle.py promote --mlflow-uri ./mlruns --from-env staging --to-env production

# Transition model to production stage
python src/model_lifecycle.py transition --mlflow-uri ./mlruns --environment production --version 1 --stage Production

# Clean up old runs (keep last 10)
python src/model_lifecycle.py cleanup --mlflow-uri ./mlruns --environment development --keep-last 10 --confirm
```

### MLflow UI Access

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5001

# Access at: http://localhost:5001
```

### Rich Visualizations Generated

Each training run automatically generates:
- `enhanced_confusion_matrix.png` - Confusion matrix with counts and percentages
- `roc_curves.png` - ROC curves for each class with AUC scores
- `enhanced_feature_importance.png` - Feature importance with error bars and pie chart
- `cv_scores.png` - Cross-validation score analysis
- `prediction_confidence.png` - Prediction confidence distributions
- `data_distribution.png` - Feature distributions by class

### Model Registry Features

- **Environment-specific registries**: `iris-classifier_development`, `iris-classifier_staging`, `iris-classifier_production`
- **Automatic versioning**: Sequential version numbers per environment
- **Stage management**: None → Staging → Production → Archived
- **Model signatures**: Automatic input/output schema inference
- **Rich metadata**: Git commits, environment info, training parameters

## 📊 MLOps Lifecycle Components

### 1. Data Processing
- **Component**: `load_and_preprocess_data`
- **Function**: Loads iris dataset, splits, and scales features
- **Outputs**: Preprocessed datasets and scaler

### 2. Model Training
- **Component**: `train_model`
- **Function**: Trains RandomForest classifier with MLflow tracking
- **Outputs**: Trained model and training metrics

### 3. Model Evaluation
- **Component**: `evaluate_model`
- **Function**: Evaluates model performance and generates reports
- **Outputs**: Evaluation metrics and visualizations

### 4. Model Serving
- **FastAPI Service**: Production-ready REST API for model inference
- **Features**: 
  - Single and batch predictions
  - Health checks and model info endpoints
  - Automatic model loading from MLflow or local files
  - Input validation and error handling
  - OpenAPI documentation

### 5. Enhanced MLflow Tracking
- **Multi-Environment Support**:
  - Development (`iris-classification-dev`)
  - Staging (`iris-classification-staging`)
  - Production (`iris-classification-prod`)
- **Comprehensive Metrics** (70+ per run):
  - Core metrics (accuracy, precision, recall, F1, AUC)
  - Cross-validation statistics (mean, std, min, max)
  - Feature importance with standard deviations
  - Per-class performance metrics
  - Prediction confidence analysis
  - Data distribution statistics
- **Advanced Visualizations**:
  - Enhanced confusion matrices (counts + percentages)
  - ROC curves for each class
  - Feature importance plots (bar + pie charts)
  - Cross-validation score analysis
  - Prediction confidence distributions
  - Data distribution by class
- **Model Lifecycle Management**:
  - Environment-specific model registries
  - Version tracking and promotion workflows
  - Stage management (Staging/Production/Archived)
  - Performance comparison across environments
  - Git integration with commit tracking
- **Enterprise Features**:
  - Model signatures and input examples
  - Detailed classification reports
  - Model interpretation documentation
  - Automated cleanup utilities

## 🔧 Configuration

### Environment Variables

```bash
# Required for GCP deployment
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1

# MLflow configuration
MLFLOW_TRACKING_URI=http://mlflow-service:5000
MLFLOW_EXPERIMENT_NAME=iris-classification

# Model parameters
TEST_SIZE=0.2
RANDOM_STATE=42
N_ESTIMATORS=100
```

### Custom Configuration

Modify `src/config.py` to add custom parameters:

```python
@dataclass
class MLOpsConfig:
    # Add your custom parameters here
    custom_param: str = "default_value"
```

## 📈 Monitoring and Observability

### Viewing Logs

```bash
# MLflow server logs
kubectl logs -n mlops-pipeline deployment/mlflow-server

# Training job logs
kubectl logs -n mlops-pipeline job/iris-training-job

# Follow logs in real-time
kubectl logs -f -n mlops-pipeline job/iris-training-job
```

### Checking Status

```bash
# Check all resources
kubectl get all -n mlops-pipeline

# Check job status
kubectl describe job iris-training-job -n mlops-pipeline

# Check persistent volumes
kubectl get pv,pvc -n mlops-pipeline
```

## 🧹 Cleanup

### Local Cleanup

```bash
# Stop Docker Compose
docker-compose -f docker/docker-compose.yml down -v

# Remove virtual environment
rm -rf venv
```

### GCP Cleanup

```bash
# Delete Kubernetes resources
kubectl delete namespace mlops-pipeline

# Delete GKE cluster
gcloud container clusters delete $CLUSTER_NAME --region=$GOOGLE_CLOUD_REGION

# Delete service account
gcloud iam service-accounts delete mlops-pipeline-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com

# Delete GCS bucket
gsutil rm -r gs://${GOOGLE_CLOUD_PROJECT}-mlflow-artifacts
```

## 🛠️ Troubleshooting

### Common Issues

1. **Pod ImagePullError**
   ```bash
   # Check if images are pushed to GCR
   gcloud container images list --repository=gcr.io/$GOOGLE_CLOUD_PROJECT
   
   # Verify image tags in manifests
   kubectl describe pod <pod-name> -n mlops-pipeline
   ```

2. **Service Account Permissions**
   ```bash
   # Check service account exists
   gcloud iam service-accounts list
   
   # Verify IAM bindings
   gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
   ```

3. **MLflow Connection Issues**
   ```bash
   # Check MLflow service
   kubectl get svc -n mlops-pipeline
   
   # Test connectivity
   kubectl exec -it <pod-name> -n mlops-pipeline -- curl http://mlflow-service:5000/health
   ```

### Debug Commands

```bash
# Get pod details
kubectl describe pod <pod-name> -n mlops-pipeline

# Access pod shell
kubectl exec -it <pod-name> -n mlops-pipeline -- /bin/bash

# Check events
kubectl get events -n mlops-pipeline --sort-by='.lastTimestamp'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and on GCP
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [Google Kubernetes Engine Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Docker Documentation](https://docs.docker.com/)

## 💡 Next Steps

- Add model serving endpoint
- Implement A/B testing
- Add data drift detection
- Set up automated retraining
- Implement model governance
- Add more sophisticated monitoring