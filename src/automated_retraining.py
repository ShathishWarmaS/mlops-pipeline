#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline for MLOps
Intelligent retraining system with drift detection, performance monitoring, and automated deployment
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import schedule
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from drift_detector import DriftDetectionPipeline, DriftReport
from monitoring import MLOpsMonitor, ModelMetrics
from ab_testing import ABTestExperiment, ExperimentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrainingTrigger:
    """Retraining trigger configuration"""
    name: str
    trigger_type: str  # drift, performance, schedule, manual
    threshold: float
    enabled: bool
    priority: int  # 1=highest, 5=lowest
    conditions: Dict[str, Any]
    
@dataclass
class RetrainingJob:
    """Retraining job definition"""
    job_id: str
    trigger: RetrainingTrigger
    model_name: str
    training_data_path: str
    validation_data_path: str
    model_config: Dict[str, Any]
    created_at: datetime
    status: str  # pending, running, completed, failed
    priority: int
    estimated_duration: timedelta
    
@dataclass
class RetrainingResult:
    """Results from a retraining job"""
    job_id: str
    model_name: str
    old_model_version: str
    new_model_version: str
    performance_improvement: float
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    training_duration: timedelta
    deploy_recommendation: str
    created_at: datetime
    artifacts: List[str]

class DataManager:
    """Manage training data collection and preparation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources = config.get('data_sources', {})
        self.data_quality_checks = config.get('data_quality', {})
        
    def collect_training_data(self, 
                            start_date: datetime, 
                            end_date: datetime,
                            include_recent: bool = True) -> pd.DataFrame:
        """Collect training data from configured sources"""
        logger.info(f"Collecting training data from {start_date} to {end_date}")
        
        # In a real implementation, this would connect to various data sources
        # For demo, we'll simulate data collection
        
        data_frames = []
        
        # Simulate database collection
        if 'database' in self.data_sources:
            db_data = self._collect_from_database(start_date, end_date)
            if db_data is not None:
                data_frames.append(db_data)
                
        # Simulate file-based collection
        if 'files' in self.data_sources:
            file_data = self._collect_from_files(start_date, end_date)
            if file_data is not None:
                data_frames.append(file_data)
                
        # Simulate streaming data collection
        if 'streaming' in self.data_sources:
            stream_data = self._collect_from_streams(start_date, end_date)
            if stream_data is not None:
                data_frames.append(stream_data)
                
        if not data_frames:
            raise ValueError("No training data collected from any source")
            
        # Combine all data sources
        combined_data = pd.concat(data_frames, ignore_index=True)
        
        # Apply data quality checks
        cleaned_data = self._apply_data_quality_checks(combined_data)
        
        logger.info(f"Collected {len(cleaned_data)} training samples")
        return cleaned_data
        
    def _collect_from_database(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Simulate database data collection"""
        # In practice, this would execute SQL queries
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        data['timestamp'] = pd.date_range(start_date, end_date, periods=len(data))
        return data
        
    def _collect_from_files(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Simulate file-based data collection"""
        # In practice, this would read from various file formats
        return None
        
    def _collect_from_streams(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Simulate streaming data collection"""
        # In practice, this would connect to Kafka, Kinesis, etc.
        return None
        
    def _apply_data_quality_checks(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality checks and cleaning"""
        logger.info("Applying data quality checks")
        
        initial_size = len(data)
        
        # Remove duplicates
        if self.data_quality_checks.get('remove_duplicates', True):
            data = data.drop_duplicates()
            
        # Handle missing values
        missing_threshold = self.data_quality_checks.get('missing_threshold', 0.1)
        for column in data.columns:
            missing_ratio = data[column].isnull().sum() / len(data)
            if missing_ratio > missing_threshold:
                logger.warning(f"Column {column} has {missing_ratio:.2%} missing values")
                
        # Remove rows with too many missing values
        row_missing_threshold = self.data_quality_checks.get('row_missing_threshold', 0.2)
        data = data.dropna(thresh=int(len(data.columns) * (1 - row_missing_threshold)))
        
        # Outlier detection (simple IQR method)
        if self.data_quality_checks.get('remove_outliers', True):
            numerical_columns = data.select_dtypes(include=[np.number]).columns
            for column in numerical_columns:
                if column != 'target':  # Don't remove outliers from target
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
                    
        final_size = len(data)
        removed_ratio = (initial_size - final_size) / initial_size
        
        logger.info(f"Data quality checks completed. Removed {removed_ratio:.2%} of data")
        return data

class ModelTrainer:
    """Advanced model training with hyperparameter optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hyperparameter_search = config.get('hyperparameter_search', {})
        
    def train_model(self, 
                   training_data: pd.DataFrame,
                   validation_data: pd.DataFrame,
                   model_config: Dict[str, Any],
                   experiment_name: str) -> Tuple[BaseEstimator, Dict[str, float]]:
        """Train model with hyperparameter optimization"""
        logger.info(f"Starting model training for {experiment_name}")
        
        # Prepare data
        feature_columns = [col for col in training_data.columns if col not in ['target', 'timestamp']]
        X_train = training_data[feature_columns]
        y_train = training_data['target']
        X_val = validation_data[feature_columns]
        y_val = validation_data['target']
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"retraining_{experiment_name}_{int(time.time())}"):
            # Log training configuration
            mlflow.log_params(model_config)
            mlflow.log_param("training_samples", len(training_data))
            mlflow.log_param("validation_samples", len(validation_data))
            
            # Hyperparameter optimization
            if self.hyperparameter_search.get('enabled', False):
                best_model, best_params = self._hyperparameter_optimization(
                    X_train, y_train, X_val, y_val, model_config
                )
                mlflow.log_params(best_params)
            else:
                # Use default parameters
                best_model = self._create_model(model_config)
                best_model.fit(X_train, y_train)
                
            # Evaluate model
            train_predictions = best_model.predict(X_train)
            val_predictions = best_model.predict(X_val)
            
            train_metrics = self._calculate_metrics(y_train, train_predictions)
            val_metrics = self._calculate_metrics(y_val, val_predictions)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            for metric, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
                
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
            mlflow.log_metric("cv_std_accuracy", cv_scores.std())
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            logger.info(f"Model training completed. Validation accuracy: {val_metrics['accuracy']:.4f}")
            
            return best_model, val_metrics
            
    def _hyperparameter_optimization(self, 
                                   X_train: pd.DataFrame, 
                                   y_train: pd.Series,
                                   X_val: pd.DataFrame, 
                                   y_val: pd.Series,
                                   model_config: Dict[str, Any]) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Perform hyperparameter optimization"""
        from sklearn.model_selection import RandomizedSearchCV
        
        # Create base model
        base_model = self._create_model(model_config)
        
        # Define parameter grid
        param_grid = self.hyperparameter_search.get('param_grid', {})
        
        if not param_grid:
            logger.warning("No parameter grid defined for hyperparameter search")
            base_model.fit(X_train, y_train)
            return base_model, {}
            
        # Perform randomized search
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=self.hyperparameter_search.get('n_iter', 20),
            cv=self.hyperparameter_search.get('cv_folds', 3),
            scoring=self.hyperparameter_search.get('scoring', 'accuracy'),
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best hyperparameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
        
    def _create_model(self, model_config: Dict[str, Any]) -> BaseEstimator:
        """Create model instance based on configuration"""
        model_type = model_config.get('type', 'RandomForestClassifier')
        model_params = model_config.get('params', {})
        
        if model_type == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**model_params)
        elif model_type == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**model_params)
        elif model_type == 'SVC':
            from sklearn.svm import SVC
            return SVC(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

class RetrainingOrchestrator:
    """Main orchestrator for automated retraining pipeline"""
    
    def __init__(self, config_path: str = "retraining_config.json"):
        self.config = self._load_config(config_path)
        self.data_manager = DataManager(self.config['data_management'])
        self.model_trainer = ModelTrainer(self.config['training'])
        self.drift_detector = DriftDetectionPipeline(self.config.get('drift_detection', {}))
        self.monitor = MLOpsMonitor()
        
        # Job management
        self.job_queue = []
        self.active_jobs = {}
        self.completed_jobs = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_concurrent_jobs', 2))
        
        # Triggers
        self.triggers = [RetrainingTrigger(**trigger) for trigger in self.config['triggers']]
        self.running = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load retraining configuration"""
        default_config = {
            "monitoring_interval_minutes": 60,
            "max_concurrent_jobs": 2,
            "data_retention_days": 90,
            "model_comparison_threshold": 0.02,
            "auto_deploy_threshold": 0.05,
            "triggers": [
                {
                    "name": "drift_trigger",
                    "trigger_type": "drift",
                    "threshold": 0.1,
                    "enabled": True,
                    "priority": 1,
                    "conditions": {
                        "min_samples": 1000,
                        "drift_features_threshold": 3
                    }
                },
                {
                    "name": "performance_trigger",
                    "trigger_type": "performance",
                    "threshold": 0.05,
                    "enabled": True,
                    "priority": 2,
                    "conditions": {
                        "metric": "accuracy",
                        "window_hours": 24,
                        "min_predictions": 100
                    }
                },
                {
                    "name": "scheduled_trigger",
                    "trigger_type": "schedule",
                    "threshold": 0.0,
                    "enabled": True,
                    "priority": 3,
                    "conditions": {
                        "frequency": "weekly",
                        "day_of_week": "sunday",
                        "hour": 2
                    }
                }
            ],
            "data_management": {
                "data_sources": {
                    "database": {"enabled": True},
                    "files": {"enabled": False},
                    "streaming": {"enabled": False}
                },
                "data_quality": {
                    "remove_duplicates": True,
                    "missing_threshold": 0.1,
                    "row_missing_threshold": 0.2,
                    "remove_outliers": True
                }
            },
            "training": {
                "hyperparameter_search": {
                    "enabled": True,
                    "n_iter": 20,
                    "cv_folds": 3,
                    "scoring": "accuracy",
                    "param_grid": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                }
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
        return default_config
        
    def start_monitoring(self):
        """Start the automated monitoring and retraining system"""
        self.running = True
        logger.info("Starting automated retraining system")
        
        # Schedule monitoring checks
        schedule.every(self.config['monitoring_interval_minutes']).minutes.do(self._check_triggers)
        
        # Schedule periodic triggers
        for trigger in self.triggers:
            if trigger.trigger_type == 'schedule' and trigger.enabled:
                self._schedule_periodic_trigger(trigger)
                
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Start job processing thread
        processing_thread = threading.Thread(target=self._process_jobs)
        processing_thread.daemon = True
        processing_thread.start()
        
        logger.info("Automated retraining system started")
        
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        schedule.clear()
        self.executor.shutdown(wait=True)
        logger.info("Automated retraining system stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
                
    def _schedule_periodic_trigger(self, trigger: RetrainingTrigger):
        """Schedule periodic retraining triggers"""
        conditions = trigger.conditions
        frequency = conditions.get('frequency', 'weekly')
        
        if frequency == 'daily':
            hour = conditions.get('hour', 2)
            schedule.every().day.at(f"{hour:02d}:00").do(
                self._trigger_retraining, trigger
            )
        elif frequency == 'weekly':
            day = conditions.get('day_of_week', 'sunday')
            hour = conditions.get('hour', 2)
            getattr(schedule.every(), day).at(f"{hour:02d}:00").do(
                self._trigger_retraining, trigger
            )
        elif frequency == 'monthly':
            # Schedule for first day of month
            schedule.every().month.do(self._trigger_retraining, trigger)
            
    def _check_triggers(self):
        """Check all enabled triggers for retraining conditions"""
        logger.info("Checking retraining triggers")
        
        for trigger in self.triggers:
            if not trigger.enabled:
                continue
                
            try:
                if trigger.trigger_type == 'drift':
                    self._check_drift_trigger(trigger)
                elif trigger.trigger_type == 'performance':
                    self._check_performance_trigger(trigger)
                # Schedule triggers are handled separately
                    
            except Exception as e:
                logger.error(f"Error checking trigger {trigger.name}: {e}")
                
    def _check_drift_trigger(self, trigger: RetrainingTrigger):
        """Check data drift trigger conditions"""
        # Get recent data for drift detection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        try:
            current_data = self.data_manager.collect_training_data(start_date, end_date)
            
            if len(current_data) < trigger.conditions.get('min_samples', 1000):
                logger.info(f"Insufficient samples for drift detection: {len(current_data)}")
                return
                
            # Check if drift detector is initialized
            if not hasattr(self.drift_detector, 'reference_data'):
                # Initialize with historical data
                historical_end = start_date
                historical_start = historical_end - timedelta(days=30)
                reference_data = self.data_manager.collect_training_data(historical_start, historical_end)
                self.drift_detector.fit_reference(reference_data)
                
            # Detect drift
            drift_report = self.drift_detector.detect_and_analyze(current_data)
            
            # Check trigger conditions
            if (drift_report.overall_drift_score > trigger.threshold and
                len(drift_report.affected_features) >= trigger.conditions.get('drift_features_threshold', 3)):
                
                logger.warning(f"Drift trigger activated: {drift_report.overall_drift_score:.3f}")
                self._trigger_retraining(trigger, drift_report=drift_report)
                
        except Exception as e:
            logger.error(f"Error in drift trigger check: {e}")
            
    def _check_performance_trigger(self, trigger: RetrainingTrigger):
        """Check model performance trigger conditions"""
        conditions = trigger.conditions
        metric = conditions.get('metric', 'accuracy')
        window_hours = conditions.get('window_hours', 24)
        min_predictions = conditions.get('min_predictions', 100)
        
        # Get recent model performance
        # This would typically query your model monitoring system
        # For demo, we'll simulate performance degradation
        
        # Simulate getting current performance
        current_performance = 0.82  # Simulated current accuracy
        baseline_performance = 0.87  # Baseline accuracy
        
        performance_drop = baseline_performance - current_performance
        
        if performance_drop > trigger.threshold:
            logger.warning(f"Performance trigger activated: {performance_drop:.3f} drop in {metric}")
            self._trigger_retraining(trigger, performance_drop=performance_drop)
            
    def _trigger_retraining(self, trigger: RetrainingTrigger, **kwargs):
        """Trigger a retraining job"""
        job_id = f"retrain_{trigger.name}_{int(time.time())}"
        
        # Collect training data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days for training
        
        job = RetrainingJob(
            job_id=job_id,
            trigger=trigger,
            model_name="iris_classifier",  # This would be configurable
            training_data_path="",  # Will be populated during execution
            validation_data_path="",
            model_config=self.config.get('model_config', {
                'type': 'RandomForestClassifier',
                'params': {'n_estimators': 100, 'random_state': 42}
            }),
            created_at=datetime.now(),
            status="pending",
            priority=trigger.priority,
            estimated_duration=timedelta(minutes=30)
        )
        
        # Add metadata from trigger
        for key, value in kwargs.items():
            setattr(job, key, value)
            
        self.job_queue.append(job)
        self.job_queue.sort(key=lambda x: x.priority)  # Sort by priority
        
        logger.info(f"Retraining job {job_id} queued with priority {trigger.priority}")
        
    def _process_jobs(self):
        """Process retraining jobs from the queue"""
        while self.running:
            try:
                if (self.job_queue and 
                    len(self.active_jobs) < self.config['max_concurrent_jobs']):
                    
                    job = self.job_queue.pop(0)
                    self.active_jobs[job.job_id] = job
                    
                    # Submit job to executor
                    future = self.executor.submit(self._execute_retraining_job, job)
                    future.add_done_callback(lambda f, job_id=job.job_id: self._job_completed(job_id, f))
                    
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in job processing: {e}")
                time.sleep(30)
                
    def _execute_retraining_job(self, job: RetrainingJob) -> RetrainingResult:
        """Execute a retraining job"""
        logger.info(f"Starting retraining job {job.job_id}")
        job.status = "running"
        start_time = datetime.now()
        
        try:
            # Collect training data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            training_data = self.data_manager.collect_training_data(start_date, end_date)
            
            # Split data
            train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
            
            # Train new model
            new_model, val_metrics = self.model_trainer.train_model(
                train_data, val_data, job.model_config, job.job_id
            )
            
            # Compare with existing model if available
            performance_improvement = self._compare_with_existing_model(
                new_model, val_data, val_metrics
            )
            
            # Generate deployment recommendation
            deploy_recommendation = self._generate_deployment_recommendation(
                performance_improvement, val_metrics
            )
            
            # Save new model
            model_version = f"v{int(time.time())}"
            model_path = f"models/{job.model_name}_{model_version}.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(new_model, model_path)
            
            training_duration = datetime.now() - start_time
            
            result = RetrainingResult(
                job_id=job.job_id,
                model_name=job.model_name,
                old_model_version="current",
                new_model_version=model_version,
                performance_improvement=performance_improvement,
                training_metrics={},  # Would include training metrics
                validation_metrics=val_metrics,
                training_duration=training_duration,
                deploy_recommendation=deploy_recommendation,
                created_at=datetime.now(),
                artifacts=[model_path]
            )
            
            job.status = "completed"
            logger.info(f"Retraining job {job.job_id} completed successfully")
            
            return result
            
        except Exception as e:
            job.status = "failed"
            logger.error(f"Retraining job {job.job_id} failed: {e}")
            raise
            
    def _compare_with_existing_model(self, 
                                   new_model: BaseEstimator,
                                   validation_data: pd.DataFrame,
                                   new_metrics: Dict[str, float]) -> float:
        """Compare new model with existing model"""
        try:
            # Load existing model (in practice, this would be from model registry)
            existing_model_path = "models/iris_classifier_current.joblib"
            if os.path.exists(existing_model_path):
                existing_model = joblib.load(existing_model_path)
                
                # Evaluate existing model on validation data
                feature_columns = [col for col in validation_data.columns if col not in ['target', 'timestamp']]
                X_val = validation_data[feature_columns]
                y_val = validation_data['target']
                
                existing_predictions = existing_model.predict(X_val)
                existing_accuracy = accuracy_score(y_val, existing_predictions)
                
                improvement = new_metrics['accuracy'] - existing_accuracy
                logger.info(f"Model improvement: {improvement:.4f}")
                return improvement
            else:
                logger.info("No existing model found for comparison")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error comparing models: {e}")
            return 0.0
            
    def _generate_deployment_recommendation(self, 
                                          improvement: float,
                                          metrics: Dict[str, float]) -> str:
        """Generate deployment recommendation based on model performance"""
        min_improvement = self.config.get('model_comparison_threshold', 0.02)
        auto_deploy_threshold = self.config.get('auto_deploy_threshold', 0.05)
        
        if improvement >= auto_deploy_threshold:
            return "AUTO_DEPLOY: Significant improvement detected"
        elif improvement >= min_improvement:
            return "MANUAL_REVIEW: Moderate improvement, manual review recommended"
        elif improvement > 0:
            return "HOLD: Minor improvement, not worth deployment"
        else:
            return "REJECT: No improvement or performance degradation"
            
    def _job_completed(self, job_id: str, future):
        """Handle job completion"""
        job = self.active_jobs.pop(job_id, None)
        if job:
            try:
                result = future.result()
                self.completed_jobs.append(result)
                
                # Log to MLflow
                self._log_retraining_result(result)
                
                # Auto-deploy if recommended
                if "AUTO_DEPLOY" in result.deploy_recommendation:
                    self._auto_deploy_model(result)
                    
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                
    def _log_retraining_result(self, result: RetrainingResult):
        """Log retraining results to MLflow"""
        with mlflow.start_run(run_name=f"retraining_result_{result.job_id}"):
            # Log metrics
            for metric, value in result.validation_metrics.items():
                mlflow.log_metric(f"val_{metric}", value)
                
            mlflow.log_metric("performance_improvement", result.performance_improvement)
            mlflow.log_metric("training_duration_minutes", result.training_duration.total_seconds() / 60)
            
            # Log parameters
            mlflow.log_param("model_name", result.model_name)
            mlflow.log_param("new_model_version", result.new_model_version)
            mlflow.log_param("deploy_recommendation", result.deploy_recommendation)
            
            # Log artifacts
            for artifact in result.artifacts:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)
                    
            mlflow.set_tag("retraining_job_id", result.job_id)
            mlflow.set_tag("retraining_timestamp", result.created_at.isoformat())
            
    def _auto_deploy_model(self, result: RetrainingResult):
        """Automatically deploy model if conditions are met"""
        logger.info(f"Auto-deploying model {result.new_model_version}")
        
        # In practice, this would:
        # 1. Update model registry
        # 2. Deploy to serving infrastructure
        # 3. Update routing configuration
        # 4. Run smoke tests
        
        # For demo, we'll just copy the model to the current model location
        source_path = result.artifacts[0]  # First artifact should be the model
        target_path = f"models/{result.model_name}_current.joblib"
        
        import shutil
        shutil.copy2(source_path, target_path)
        
        logger.info(f"Model {result.new_model_version} deployed successfully")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.running,
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "completed_jobs": len(self.completed_jobs),
            "enabled_triggers": [t.name for t in self.triggers if t.enabled],
            "last_check": datetime.now().isoformat()
        }

def create_retraining_config():
    """Create sample retraining configuration"""
    config = {
        "monitoring_interval_minutes": 60,
        "max_concurrent_jobs": 2,
        "data_retention_days": 90,
        "model_comparison_threshold": 0.02,
        "auto_deploy_threshold": 0.05,
        "triggers": [
            {
                "name": "drift_trigger",
                "trigger_type": "drift",
                "threshold": 0.1,
                "enabled": True,
                "priority": 1,
                "conditions": {
                    "min_samples": 1000,
                    "drift_features_threshold": 3
                }
            },
            {
                "name": "performance_trigger",
                "trigger_type": "performance",
                "threshold": 0.05,
                "enabled": True,
                "priority": 2,
                "conditions": {
                    "metric": "accuracy",
                    "window_hours": 24,
                    "min_predictions": 100
                }
            },
            {
                "name": "scheduled_trigger",
                "trigger_type": "schedule",
                "threshold": 0.0,
                "enabled": True,
                "priority": 3,
                "conditions": {
                    "frequency": "weekly",
                    "day_of_week": "sunday",
                    "hour": 2
                }
            }
        ],
        "model_config": {
            "type": "RandomForestClassifier",
            "params": {
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "data_management": {
            "data_sources": {
                "database": {"enabled": True},
                "files": {"enabled": False},
                "streaming": {"enabled": False}
            },
            "data_quality": {
                "remove_duplicates": True,
                "missing_threshold": 0.1,
                "row_missing_threshold": 0.2,
                "remove_outliers": True
            }
        },
        "training": {
            "hyperparameter_search": {
                "enabled": True,
                "n_iter": 20,
                "cv_folds": 3,
                "scoring": "accuracy",
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        }
    }
    
    with open("retraining_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Created retraining_config.json")

if __name__ == "__main__":
    # Create sample configuration
    create_retraining_config()
    
    # Example usage
    orchestrator = RetrainingOrchestrator()
    
    # Start monitoring (in practice, this would run as a service)
    print("Starting retraining orchestrator...")
    orchestrator.start_monitoring()
    
    # Simulate running for a short time
    time.sleep(5)
    
    # Get status
    status = orchestrator.get_system_status()
    print(f"System status: {status}")
    
    # Stop monitoring
    orchestrator.stop_monitoring()
    print("Retraining orchestrator stopped")