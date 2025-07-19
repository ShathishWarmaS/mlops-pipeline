"""Enhanced MLflow manager for complete model lifecycle management."""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from config import MLOpsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """Enhanced MLflow manager for dev/staging/prod lifecycle."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        # Initialize experiments for different environments
        self.environments = {
            "development": f"{config.mlflow_experiment_name}-dev",
            "staging": f"{config.mlflow_experiment_name}-staging", 
            "production": f"{config.mlflow_experiment_name}-prod"
        }
        
        self._setup_experiments()
    
    def _setup_experiments(self):
        """Setup experiments for different environments."""
        for env, exp_name in self.environments.items():
            try:
                experiment = self.client.get_experiment_by_name(exp_name)
                if experiment is None:
                    exp_id = self.client.create_experiment(
                        name=exp_name,
                        tags={
                            "environment": env,
                            "project": "iris-classification",
                            "team": "ml-team",
                            "created_date": datetime.now().isoformat()
                        }
                    )
                    logger.info(f"Created experiment '{exp_name}' with ID: {exp_id}")
                else:
                    logger.info(f"Experiment '{exp_name}' already exists")
            except Exception as e:
                logger.error(f"Error setting up experiment {exp_name}: {e}")
    
    def start_run(self, environment: str = "development", 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run with enhanced metadata."""
        
        if environment not in self.environments:
            raise ValueError(f"Environment must be one of: {list(self.environments.keys())}")
        
        experiment_name = self.environments[environment]
        mlflow.set_experiment(experiment_name)
        
        # Default run name with timestamp
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"iris_classification_{timestamp}"
        
        # Enhanced default tags
        default_tags = {
            "environment": environment,
            "model_type": "RandomForestClassifier",
            "dataset": "iris",
            "framework": "scikit-learn",
            "pipeline_version": "1.0.0",
            "git_commit": self._get_git_commit(),
            "created_by": os.getenv("USER", "unknown"),
            "run_date": datetime.now().isoformat()
        }
        
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started run '{run_name}' in environment '{environment}'")
        
        return run
    
    def log_model_details(self, model, model_name: str, environment: str = "development"):
        """Log comprehensive model details."""
        
        # Model parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param, value in params.items():
                mlflow.log_param(f"model_{param}", value)
        
        # Model metadata
        model_info = {
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "environment": environment,
            "training_timestamp": datetime.now().isoformat()
        }
        
        # Log as parameters and artifacts
        for key, value in model_info.items():
            mlflow.log_param(key, value)
        
        # Save model info as artifact
        model_info_path = "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        mlflow.log_artifact(model_info_path)
        os.remove(model_info_path)
        
        # Log model to registry
        model_uri = mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"{model_name}_{environment}",
            signature=self._infer_signature(),
            input_example=self._get_input_example()
        ).model_uri
        
        logger.info(f"Model logged to registry: {model_uri}")
        return model_uri
    
    def log_training_metrics(self, metrics: Dict[str, float], environment: str = "development"):
        """Log comprehensive training metrics."""
        
        # Core metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Additional computed metrics
        if 'train_accuracy' in metrics and 'test_accuracy' in metrics:
            overfitting_score = metrics['train_accuracy'] - metrics['test_accuracy']
            mlflow.log_metric("overfitting_score", overfitting_score)
            
            # Performance category
            if metrics['test_accuracy'] >= 0.95:
                performance_category = "excellent"
            elif metrics['test_accuracy'] >= 0.9:
                performance_category = "good"
            elif metrics['test_accuracy'] >= 0.8:
                performance_category = "acceptable"
            else:
                performance_category = "poor"
            
            mlflow.log_param("performance_category", performance_category)
        
        # Environment-specific tags
        mlflow.set_tag("metrics_environment", environment)
        mlflow.set_tag("metrics_logged_at", datetime.now().isoformat())
    
    def log_data_info(self, X_train, X_test, y_train, y_test):
        """Log dataset information and statistics."""
        
        data_info = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(X_train) + len(X_test),
            "n_features": X_train.shape[1],
            "n_classes": len(set(y_train)),
            "class_distribution_train": {str(i): int(sum(y_train == i)) for i in set(y_train)},
            "class_distribution_test": {str(i): int(sum(y_test == i)) for i in set(y_test)},
            "train_test_ratio": f"{len(X_train)}:{len(X_test)}"
        }
        
        # Log as parameters
        for key, value in data_info.items():
            if isinstance(value, dict):
                mlflow.log_param(key, json.dumps(value))
            else:
                mlflow.log_param(key, value)
        
        # Log detailed data statistics
        import numpy as np
        
        feature_stats = {}
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        for i, feature_name in enumerate(feature_names):
            stats = {
                "mean": float(np.mean(X_train[:, i])),
                "std": float(np.std(X_train[:, i])),
                "min": float(np.min(X_train[:, i])),
                "max": float(np.max(X_train[:, i])),
                "median": float(np.median(X_train[:, i]))
            }
            feature_stats[feature_name] = stats
            
            # Log individual feature statistics
            for stat_name, stat_value in stats.items():
                mlflow.log_metric(f"{feature_name}_{stat_name}", stat_value)
        
        # Save detailed stats as artifact
        stats_path = "data_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "dataset_info": data_info,
                "feature_statistics": feature_stats
            }, f, indent=2)
        mlflow.log_artifact(stats_path)
        os.remove(stats_path)
    
    def promote_model(self, model_name: str, from_env: str, to_env: str, 
                     version: Optional[str] = None) -> str:
        """Promote model from one environment to another."""
        
        from_model_name = f"{model_name}_{from_env}"
        to_model_name = f"{model_name}_{to_env}"
        
        try:
            # Get the latest version if not specified
            if version is None:
                latest_versions = self.client.get_latest_versions(
                    name=from_model_name,
                    stages=["Production", "Staging", "None"]
                )
                if not latest_versions:
                    raise ValueError(f"No versions found for model {from_model_name}")
                version = latest_versions[0].version
            
            # Get model version details
            model_version = self.client.get_model_version(from_model_name, version)
            
            # Create new model version in target environment
            result = self.client.create_model_version(
                name=to_model_name,
                source=model_version.source,
                description=f"Promoted from {from_env} environment (v{version})"
            )
            
            # Add promotion tags
            self.client.set_model_version_tag(
                name=to_model_name,
                version=result.version,
                key="promoted_from",
                value=f"{from_env}_v{version}"
            )
            
            self.client.set_model_version_tag(
                name=to_model_name,
                version=result.version,
                key="promotion_date",
                value=datetime.now().isoformat()
            )
            
            logger.info(f"Model promoted from {from_env} to {to_env}: v{result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            raise
    
    def transition_model_stage(self, model_name: str, version: str, 
                              stage: str, environment: str = "production"):
        """Transition model to different stage (Staging/Production/Archived)."""
        
        full_model_name = f"{model_name}_{environment}"
        
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of: {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=full_model_name,
                version=version,
                stage=stage
            )
            
            # Add stage transition tag
            self.client.set_model_version_tag(
                name=full_model_name,
                version=version,
                key="stage_transition_date",
                value=datetime.now().isoformat()
            )
            
            logger.info(f"Model {full_model_name} v{version} transitioned to {stage}")
            
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise
    
    def get_model_performance_comparison(self, model_name: str) -> Dict[str, Any]:
        """Compare model performance across environments."""
        
        comparison = {}
        
        for env in self.environments.keys():
            try:
                full_model_name = f"{model_name}_{env}"
                latest_versions = self.client.get_latest_versions(
                    name=full_model_name,
                    stages=["Production", "Staging", "None"]
                )
                
                if latest_versions:
                    version = latest_versions[0]
                    
                    # Get the run associated with this model version
                    run = self.client.get_run(version.run_id)
                    
                    comparison[env] = {
                        "version": version.version,
                        "stage": version.current_stage,
                        "metrics": run.data.metrics,
                        "created_date": version.creation_timestamp,
                        "run_id": version.run_id
                    }
                    
            except Exception as e:
                logger.warning(f"Could not get performance for {env}: {e}")
                comparison[env] = {"error": str(e)}
        
        return comparison
    
    def get_production_model(self, model_name: str):
        """Get the current production model."""
        
        prod_model_name = f"{model_name}_production"
        
        try:
            # Get production models
            prod_models = self.client.get_latest_versions(
                name=prod_model_name,
                stages=["Production"]
            )
            
            if not prod_models:
                logger.warning(f"No production models found for {prod_model_name}")
                return None
            
            model_version = prod_models[0]
            
            # Load the model
            model_uri = f"models:/{prod_model_name}/{model_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            logger.info(f"Loaded production model: {prod_model_name} v{model_version.version}")
            
            return {
                "model": model,
                "version": model_version.version,
                "model_uri": model_uri,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id
            }
            
        except Exception as e:
            logger.error(f"Error loading production model: {e}")
            return None
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True,
                cwd=os.path.dirname(__file__)
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _infer_signature(self):
        """Infer model signature for MLflow."""
        try:
            import numpy as np
            from mlflow.models.signature import infer_signature
            
            # Sample input and output for signature
            sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
            sample_output = np.array([0])  # Sample prediction
            
            return infer_signature(sample_input, sample_output)
        except Exception as e:
            logger.warning(f"Could not infer signature: {e}")
            return None
    
    def _get_input_example(self):
        """Get input example for MLflow."""
        try:
            import numpy as np
            return np.array([[5.1, 3.5, 1.4, 0.2]])
        except:
            return None
    
    def cleanup_old_runs(self, environment: str, keep_last_n: int = 10):
        """Clean up old experiment runs to save space."""
        
        experiment_name = self.environments[environment]
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.warning(f"Experiment {experiment_name} not found")
            return
        
        # Get all runs for the experiment
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["start_time DESC"]
        )
        
        # Delete runs beyond the keep limit
        if len(runs) > keep_last_n:
            runs_to_delete = runs[keep_last_n:]
            
            for run in runs_to_delete:
                try:
                    self.client.delete_run(run.info.run_id)
                    logger.info(f"Deleted old run: {run.info.run_id}")
                except Exception as e:
                    logger.warning(f"Could not delete run {run.info.run_id}: {e}")
            
            logger.info(f"Cleaned up {len(runs_to_delete)} old runs from {environment}")
    
    def generate_model_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive model report across all environments."""
        
        report = {
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
            "environments": {},
            "performance_comparison": self.get_model_performance_comparison(model_name)
        }
        
        for env in self.environments.keys():
            try:
                full_model_name = f"{model_name}_{env}"
                
                # Get all versions
                versions = self.client.search_model_versions(f"name='{full_model_name}'")
                
                env_info = {
                    "total_versions": len(versions),
                    "versions": []
                }
                
                for version in versions:
                    version_info = {
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "created_date": version.creation_timestamp,
                        "run_id": version.run_id
                    }
                    
                    # Get run metrics
                    try:
                        run = self.client.get_run(version.run_id)
                        version_info["metrics"] = run.data.metrics
                        version_info["parameters"] = run.data.params
                    except:
                        pass
                    
                    env_info["versions"].append(version_info)
                
                report["environments"][env] = env_info
                
            except Exception as e:
                logger.warning(f"Error generating report for {env}: {e}")
                report["environments"][env] = {"error": str(e)}
        
        return report