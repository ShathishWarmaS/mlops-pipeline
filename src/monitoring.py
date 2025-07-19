#!/usr/bin/env python3
"""
Advanced MLOps Monitoring and Alerting System
Provides comprehensive monitoring for model performance, data drift, and system health
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
import requests
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_latency: float
    throughput: float
    error_rate: float
    timestamp: datetime
    environment: str
    model_version: str

@dataclass
class DataDriftMetrics:
    """Data drift detection metrics"""
    feature_drift_scores: Dict[str, float]
    overall_drift_score: float
    drift_detected: bool
    drift_threshold: float
    timestamp: datetime
    reference_period: str

@dataclass
class SystemMetrics:
    """System health metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    error_count: int
    request_count: int
    timestamp: datetime

class PrometheusMetrics:
    """Prometheus metrics collector for MLOps monitoring"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Model performance metrics
        self.model_accuracy = Gauge('mlops_model_accuracy', 'Model accuracy score', 
                                  ['environment', 'model_version'], registry=self.registry)
        self.model_precision = Gauge('mlops_model_precision', 'Model precision score', 
                                   ['environment', 'model_version'], registry=self.registry)
        self.model_recall = Gauge('mlops_model_recall', 'Model recall score', 
                                ['environment', 'model_version'], registry=self.registry)
        self.model_f1 = Gauge('mlops_model_f1', 'Model F1 score', 
                            ['environment', 'model_version'], registry=self.registry)
        
        # Prediction metrics
        self.prediction_latency = Histogram('mlops_prediction_latency_seconds', 
                                          'Prediction latency in seconds', registry=self.registry)
        self.prediction_counter = Counter('mlops_predictions_total', 
                                        'Total number of predictions', ['environment'], registry=self.registry)
        self.error_counter = Counter('mlops_errors_total', 
                                   'Total number of errors', ['error_type'], registry=self.registry)
        
        # Data drift metrics
        self.data_drift_score = Gauge('mlops_data_drift_score', 'Overall data drift score', 
                                    ['environment'], registry=self.registry)
        self.feature_drift = Gauge('mlops_feature_drift_score', 'Feature-level drift score', 
                                 ['feature_name', 'environment'], registry=self.registry)
        
        # System metrics
        self.cpu_usage = Gauge('mlops_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('mlops_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_usage = Gauge('mlops_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        
    def update_model_metrics(self, metrics: ModelMetrics):
        """Update model performance metrics"""
        labels = [metrics.environment, metrics.model_version]
        self.model_accuracy.labels(*labels).set(metrics.accuracy)
        self.model_precision.labels(*labels).set(metrics.precision)
        self.model_recall.labels(*labels).set(metrics.recall)
        self.model_f1.labels(*labels).set(metrics.f1_score)
        
    def update_drift_metrics(self, metrics: DataDriftMetrics, environment: str):
        """Update data drift metrics"""
        self.data_drift_score.labels(environment).set(metrics.overall_drift_score)
        for feature, score in metrics.feature_drift_scores.items():
            self.feature_drift.labels(feature, environment).set(score)
            
    def update_system_metrics(self, metrics: SystemMetrics):
        """Update system health metrics"""
        self.cpu_usage.set(metrics.cpu_usage)
        self.memory_usage.set(metrics.memory_usage)
        self.disk_usage.set(metrics.disk_usage)
        
    def record_prediction(self, latency: float, environment: str):
        """Record prediction metrics"""
        self.prediction_latency.observe(latency)
        self.prediction_counter.labels(environment).inc()
        
    def record_error(self, error_type: str):
        """Record error metrics"""
        self.error_counter.labels(error_type).inc()
        
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')

class DataDriftDetector:
    """Advanced data drift detection using statistical methods"""
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.1):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.reference_stats = self._calculate_stats(reference_data)
        
    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistical properties of the data"""
        stats = {}
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                stats[column] = {
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'q25': data[column].quantile(0.25),
                    'q50': data[column].quantile(0.50),
                    'q75': data[column].quantile(0.75)
                }
        return stats
        
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        cur_props = np.where(cur_props == 0, 0.0001, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return psi
        
    def detect_drift(self, current_data: pd.DataFrame) -> DataDriftMetrics:
        """Detect data drift in current data compared to reference"""
        current_stats = self._calculate_stats(current_data)
        feature_drift_scores = {}
        
        for feature in self.reference_stats.keys():
            if feature in current_data.columns:
                # Calculate PSI for the feature
                psi = self._calculate_psi(
                    self.reference_data[feature], 
                    current_data[feature]
                )
                feature_drift_scores[feature] = psi
                
        # Overall drift score (average PSI)
        overall_drift_score = np.mean(list(feature_drift_scores.values()))
        drift_detected = overall_drift_score > self.drift_threshold
        
        return DataDriftMetrics(
            feature_drift_scores=feature_drift_scores,
            overall_drift_score=overall_drift_score,
            drift_detected=drift_detected,
            drift_threshold=self.drift_threshold,
            timestamp=datetime.now(),
            reference_period=f"{len(self.reference_data)} samples"
        )

class AlertManager:
    """Advanced alerting system for MLOps monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = []
        
    def _send_email_alert(self, subject: str, body: str, recipients: List[str]):
        """Send email alert"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['email']['sender']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'html'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], 
                                self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['username'], 
                        self.config['email']['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            
    def _send_slack_alert(self, message: str, channel: str):
        """Send Slack alert"""
        try:
            webhook_url = self.config['slack']['webhook_url']
            payload = {
                'channel': channel,
                'text': message,
                'username': 'MLOps-Monitor',
                'icon_emoji': ':warning:'
            }
            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            
    def check_model_performance_alerts(self, metrics: ModelMetrics):
        """Check for model performance degradation alerts"""
        alerts = []
        
        # Accuracy degradation
        if metrics.accuracy < self.config['thresholds']['min_accuracy']:
            alerts.append({
                'type': 'model_performance',
                'severity': 'high',
                'message': f"Model accuracy dropped to {metrics.accuracy:.3f} "
                          f"(threshold: {self.config['thresholds']['min_accuracy']})",
                'environment': metrics.environment,
                'timestamp': metrics.timestamp
            })
            
        # High error rate
        if metrics.error_rate > self.config['thresholds']['max_error_rate']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"Error rate increased to {metrics.error_rate:.3f} "
                          f"(threshold: {self.config['thresholds']['max_error_rate']})",
                'environment': metrics.environment,
                'timestamp': metrics.timestamp
            })
            
        # High latency
        if metrics.prediction_latency > self.config['thresholds']['max_latency']:
            alerts.append({
                'type': 'latency',
                'severity': 'medium',
                'message': f"Prediction latency increased to {metrics.prediction_latency:.3f}s "
                          f"(threshold: {self.config['thresholds']['max_latency']}s)",
                'environment': metrics.environment,
                'timestamp': metrics.timestamp
            })
            
        return alerts
        
    def check_drift_alerts(self, drift_metrics: DataDriftMetrics, environment: str):
        """Check for data drift alerts"""
        alerts = []
        
        if drift_metrics.drift_detected:
            alerts.append({
                'type': 'data_drift',
                'severity': 'high',
                'message': f"Data drift detected with score {drift_metrics.overall_drift_score:.3f} "
                          f"(threshold: {drift_metrics.drift_threshold})",
                'environment': environment,
                'timestamp': drift_metrics.timestamp,
                'details': drift_metrics.feature_drift_scores
            })
            
        return alerts
        
    def send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts through configured channels"""
        for alert in alerts:
            self.alert_history.append(alert)
            
            # Format alert message
            subject = f"MLOps Alert: {alert['type'].title()} - {alert['severity'].title()}"
            body = f"""
            <html>
            <body>
                <h2>MLOps Monitoring Alert</h2>
                <p><strong>Type:</strong> {alert['type']}</p>
                <p><strong>Severity:</strong> {alert['severity']}</p>
                <p><strong>Environment:</strong> {alert['environment']}</p>
                <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
                <p><strong>Message:</strong> {alert['message']}</p>
                {f"<p><strong>Details:</strong> {alert.get('details', 'N/A')}</p>" if 'details' in alert else ""}
            </body>
            </html>
            """
            
            # Send email alerts
            if 'email' in self.config and alert['severity'] in ['high', 'critical']:
                self._send_email_alert(
                    subject, 
                    body, 
                    self.config['email']['recipients']
                )
                
            # Send Slack alerts
            if 'slack' in self.config:
                slack_message = f"🚨 *{subject}*\n{alert['message']}"
                self._send_slack_alert(
                    slack_message, 
                    self.config['slack']['channel']
                )

class MLOpsMonitor:
    """Main MLOps monitoring orchestrator"""
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config = self._load_config(config_path)
        self.prometheus_metrics = PrometheusMetrics()
        self.alert_manager = AlertManager(self.config['alerting'])
        self.drift_detector = None
        self.monitoring_data = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "check_interval": 300,  # 5 minutes
                "drift_threshold": 0.1,
                "retention_days": 30
            },
            "thresholds": {
                "min_accuracy": 0.85,
                "max_error_rate": 0.05,
                "max_latency": 1.0,
                "max_drift_score": 0.1
            },
            "alerting": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender": "mlops@company.com",
                    "username": "",
                    "password": "",
                    "recipients": ["team@company.com"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#mlops-alerts"
                }
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
        return default_config
        
    def initialize_drift_detector(self, reference_data: pd.DataFrame):
        """Initialize data drift detector with reference data"""
        self.drift_detector = DataDriftDetector(
            reference_data, 
            self.config['thresholds']['max_drift_score']
        )
        logger.info("Data drift detector initialized")
        
    def evaluate_model_performance(self, model: BaseEstimator, X_test: np.ndarray, 
                                 y_test: np.ndarray, environment: str, 
                                 model_version: str) -> ModelMetrics:
        """Evaluate model performance and record metrics"""
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        prediction_latency = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate throughput and error rate
        throughput = len(X_test) / prediction_latency
        error_rate = 1 - accuracy  # Simple error rate calculation
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            prediction_latency=prediction_latency,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=datetime.now(),
            environment=environment,
            model_version=model_version
        )
        
        # Update Prometheus metrics
        self.prometheus_metrics.update_model_metrics(metrics)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"monitoring_{environment}_{int(time.time())}"):
            mlflow.log_metrics({
                "monitoring_accuracy": accuracy,
                "monitoring_precision": precision,
                "monitoring_recall": recall,
                "monitoring_f1": f1,
                "monitoring_latency": prediction_latency,
                "monitoring_throughput": throughput,
                "monitoring_error_rate": error_rate
            })
            mlflow.set_tag("monitoring_environment", environment)
            mlflow.set_tag("monitoring_timestamp", metrics.timestamp.isoformat())
            
        return metrics
        
    def monitor_data_drift(self, current_data: pd.DataFrame, environment: str) -> DataDriftMetrics:
        """Monitor data drift in current data"""
        if self.drift_detector is None:
            raise ValueError("Drift detector not initialized. Call initialize_drift_detector first.")
            
        drift_metrics = self.drift_detector.detect_drift(current_data)
        
        # Update Prometheus metrics
        self.prometheus_metrics.update_drift_metrics(drift_metrics, environment)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"drift_monitoring_{environment}_{int(time.time())}"):
            mlflow.log_metrics({
                "drift_overall_score": drift_metrics.overall_drift_score,
                "drift_detected": float(drift_metrics.drift_detected),
                **{f"drift_{feature}": score for feature, score in drift_metrics.feature_drift_scores.items()}
            })
            mlflow.set_tag("drift_environment", environment)
            mlflow.set_tag("drift_timestamp", drift_metrics.timestamp.isoformat())
            
        return drift_metrics
        
    def run_monitoring_cycle(self, model: BaseEstimator, X_test: np.ndarray, 
                           y_test: np.ndarray, current_data: pd.DataFrame, 
                           environment: str, model_version: str):
        """Run complete monitoring cycle"""
        logger.info(f"Starting monitoring cycle for {environment}")
        
        # Evaluate model performance
        model_metrics = self.evaluate_model_performance(
            model, X_test, y_test, environment, model_version
        )
        
        # Monitor data drift
        drift_metrics = self.monitor_data_drift(current_data, environment)
        
        # Check for alerts
        performance_alerts = self.alert_manager.check_model_performance_alerts(model_metrics)
        drift_alerts = self.alert_manager.check_drift_alerts(drift_metrics, environment)
        
        all_alerts = performance_alerts + drift_alerts
        
        if all_alerts:
            self.alert_manager.send_alerts(all_alerts)
            logger.warning(f"Generated {len(all_alerts)} alerts for {environment}")
        else:
            logger.info(f"No alerts generated for {environment}")
            
        # Store monitoring data
        monitoring_record = {
            'timestamp': datetime.now().isoformat(),
            'environment': environment,
            'model_version': model_version,
            'model_metrics': asdict(model_metrics),
            'drift_metrics': asdict(drift_metrics),
            'alerts': all_alerts
        }
        self.monitoring_data.append(monitoring_record)
        
        return monitoring_record
        
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        if not self.monitoring_data:
            return {"message": "No monitoring data available"}
            
        # Get latest metrics for each environment
        latest_metrics = {}
        for record in self.monitoring_data:
            env = record['environment']
            if env not in latest_metrics or record['timestamp'] > latest_metrics[env]['timestamp']:
                latest_metrics[env] = record
                
        # Calculate trends
        trends = {}
        for env in latest_metrics:
            env_data = [r for r in self.monitoring_data if r['environment'] == env]
            if len(env_data) >= 2:
                recent = env_data[-2:]
                accuracy_trend = recent[-1]['model_metrics']['accuracy'] - recent[-2]['model_metrics']['accuracy']
                drift_trend = recent[-1]['drift_metrics']['overall_drift_score'] - recent[-2]['drift_metrics']['overall_drift_score']
                trends[env] = {
                    'accuracy_trend': accuracy_trend,
                    'drift_trend': drift_trend
                }
                
        return {
            'latest_metrics': latest_metrics,
            'trends': trends,
            'total_alerts': sum(len(r['alerts']) for r in self.monitoring_data),
            'monitoring_uptime': len(self.monitoring_data),
            'prometheus_metrics': self.prometheus_metrics.get_metrics()
        }
        
    def cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_date = datetime.now() - timedelta(days=self.config['monitoring']['retention_days'])
        self.monitoring_data = [
            record for record in self.monitoring_data 
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        logger.info(f"Cleaned up monitoring data older than {self.config['monitoring']['retention_days']} days")

def create_monitoring_config():
    """Create a sample monitoring configuration file"""
    config = {
        "monitoring": {
            "check_interval": 300,
            "drift_threshold": 0.1,
            "retention_days": 30
        },
        "thresholds": {
            "min_accuracy": 0.85,
            "max_error_rate": 0.05,
            "max_latency": 1.0,
            "max_drift_score": 0.1
        },
        "alerting": {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender": "mlops@yourcompany.com",
                "username": "your-email@gmail.com",
                "password": "your-app-password",
                "recipients": ["team@yourcompany.com", "devops@yourcompany.com"]
            },
            "slack": {
                "enabled": True,
                "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "channel": "#mlops-alerts"
            }
        }
    }
    
    with open("monitoring_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created monitoring_config.json - Please update with your actual credentials")

if __name__ == "__main__":
    # Create sample configuration
    create_monitoring_config()
    
    # Example usage
    monitor = MLOpsMonitor()
    print("MLOps monitoring system initialized")
    print("Dashboard data:", monitor.get_monitoring_dashboard_data())