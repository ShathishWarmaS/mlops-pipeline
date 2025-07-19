#!/usr/bin/env python3
"""
Advanced A/B Testing and Model Performance Monitoring System
Comprehensive A/B testing framework for ML models with statistical analysis
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ABTestResult:
    """A/B test statistical analysis result"""
    test_name: str
    model_a_name: str
    model_b_name: str
    metric: str
    model_a_value: float
    model_b_value: float
    improvement: float
    improvement_percentage: float
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size_a: int
    sample_size_b: int
    power: float
    effect_size: float
    recommendation: str
    timestamp: datetime

@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    name: str
    model_a_path: str
    model_b_path: str
    traffic_split: float  # Percentage for model B (0.0 to 1.0)
    duration_days: int
    success_metrics: List[str]
    minimum_sample_size: int
    significance_level: float
    power_threshold: float
    minimum_effect_size: float
    auto_conclude: bool

class ModelPerformanceTracker:
    """Track and compare model performance metrics in real-time"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.performance_data = []
        self.model_registry = {}
        
    def register_model(self, model_name: str, model, model_version: str = "1.0"):
        """Register a model for A/B testing"""
        self.model_registry[model_name] = {
            'model': model,
            'version': model_version,
            'predictions': [],
            'latencies': [],
            'errors': [],
            'timestamps': []
        }
        logger.info(f"Registered model {model_name} v{model_version}")
        
    def record_prediction(self, 
                         model_name: str, 
                         prediction: Any, 
                         actual: Any = None,
                         latency: float = 0.0,
                         features: Dict[str, Any] = None,
                         user_id: str = None):
        """Record a prediction for performance tracking"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not registered")
            
        record = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'prediction': prediction,
            'actual': actual,
            'latency': latency,
            'features': features,
            'user_id': user_id,
            'correct': actual == prediction if actual is not None else None
        }
        
        self.performance_data.append(record)
        
        # Update model registry
        self.model_registry[model_name]['predictions'].append(prediction)
        self.model_registry[model_name]['latencies'].append(latency)
        self.model_registry[model_name]['timestamps'].append(record['timestamp'])
        
        if actual is not None:
            self.model_registry[model_name]['errors'].append(
                0 if prediction == actual else 1
            )
            
    def get_model_metrics(self, 
                         model_name: str, 
                         time_window_hours: int = 24) -> Dict[str, float]:
        """Get performance metrics for a model within time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter data for the model and time window
        model_data = [
            record for record in self.performance_data
            if record['model_name'] == model_name and record['timestamp'] > cutoff_time
        ]
        
        if not model_data:
            return {}
            
        # Calculate metrics
        predictions = [r['prediction'] for r in model_data if r['actual'] is not None]
        actuals = [r['actual'] for r in model_data if r['actual'] is not None]
        latencies = [r['latency'] for r in model_data]
        
        metrics = {
            'total_predictions': len(model_data),
            'labeled_predictions': len(predictions),
            'average_latency': np.mean(latencies) if latencies else 0.0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0.0,
            'prediction_rate': len(model_data) / time_window_hours  # per hour
        }
        
        if predictions and actuals:
            # Classification metrics
            try:
                metrics.update({
                    'accuracy': accuracy_score(actuals, predictions),
                    'precision': precision_score(actuals, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(actuals, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(actuals, predictions, average='weighted', zero_division=0)
                })
                
                # ROC AUC for binary classification
                if len(set(actuals)) == 2:
                    try:
                        metrics['roc_auc'] = roc_auc_score(actuals, predictions)
                    except ValueError:
                        pass
                        
            except Exception as e:
                logger.warning(f"Error calculating metrics for {model_name}: {e}")
                
        return metrics

class ABTestAnalyzer:
    """Statistical analysis for A/B testing of ML models"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def calculate_sample_size(self, 
                            effect_size: float, 
                            power: float = 0.8, 
                            significance_level: float = None) -> int:
        """Calculate required sample size for A/B test"""
        if significance_level is None:
            significance_level = self.significance_level
            
        # Using Cohen's formula for sample size calculation
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
        
    def statistical_test(self, 
                        values_a: List[float], 
                        values_b: List[float],
                        test_type: str = 'ttest') -> Dict[str, float]:
        """Perform statistical test between two groups"""
        if test_type == 'ttest':
            # Two-sample t-test
            statistic, p_value = ttest_ind(values_a, values_b, equal_var=False)
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            statistic, p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level
        }
        
    def calculate_confidence_interval(self, 
                                    values_a: List[float], 
                                    values_b: List[float],
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        
        # Pooled standard error
        se_a = stats.sem(values_a)
        se_b = stats.sem(values_b)
        se_diff = np.sqrt(se_a**2 + se_b**2)
        
        # Degrees of freedom (Welch's t-test)
        df = len(values_a) + len(values_b) - 2
        
        # t-critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Confidence interval
        diff = mean_b - mean_a
        margin_error = t_critical * se_diff
        
        return (diff - margin_error, diff + margin_error)
        
    def calculate_effect_size(self, values_a: List[float], values_b: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) + 
                             (len(values_b) - 1) * np.var(values_b, ddof=1)) / 
                            (len(values_a) + len(values_b) - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (mean_b - mean_a) / pooled_std
        
    def calculate_power(self, 
                       values_a: List[float], 
                       values_b: List[float],
                       significance_level: float = None) -> float:
        """Calculate statistical power of the test"""
        if significance_level is None:
            significance_level = self.significance_level
            
        effect_size = self.calculate_effect_size(values_a, values_b)
        n = min(len(values_a), len(values_b))
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n / 2)
        
        # Power calculation
        power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
        
        return power

class ABTestExperiment:
    """Complete A/B testing experiment management"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.performance_tracker = ModelPerformanceTracker(config.name)
        self.analyzer = ABTestAnalyzer(config.significance_level)
        self.start_time = datetime.now()
        self.status = "running"
        self.results = []
        
    def setup_models(self, model_a, model_b):
        """Setup models for A/B testing"""
        self.performance_tracker.register_model("model_a", model_a)
        self.performance_tracker.register_model("model_b", model_b)
        logger.info(f"A/B test experiment '{self.config.name}' started")
        
    def route_prediction(self, features: Dict[str, Any], user_id: str = None) -> Tuple[str, Any]:
        """Route prediction to model A or B based on traffic split"""
        # Simple hash-based routing for consistent user experience
        if user_id:
            hash_value = hash(user_id) % 100
            use_model_b = (hash_value / 100) < self.config.traffic_split
        else:
            # Random routing if no user ID
            use_model_b = np.random.random() < self.config.traffic_split
            
        model_name = "model_b" if use_model_b else "model_a"
        model = self.performance_tracker.model_registry[model_name]['model']
        
        # Make prediction
        start_time = datetime.now()
        prediction = model.predict([list(features.values())])[0]
        latency = (datetime.now() - start_time).total_seconds()
        
        # Record prediction
        self.performance_tracker.record_prediction(
            model_name=model_name,
            prediction=prediction,
            latency=latency,
            features=features,
            user_id=user_id
        )
        
        return model_name, prediction
        
    def update_ground_truth(self, user_id: str, actual_value: Any):
        """Update with ground truth labels for evaluation"""
        # Find the corresponding prediction
        for record in reversed(self.performance_tracker.performance_data):
            if record['user_id'] == user_id and record['actual'] is None:
                record['actual'] = actual_value
                record['correct'] = record['prediction'] == actual_value
                break
                
    def analyze_results(self) -> List[ABTestResult]:
        """Analyze A/B test results for all success metrics"""
        results = []
        
        for metric in self.config.success_metrics:
            result = self._analyze_metric(metric)
            if result:
                results.append(result)
                
        self.results = results
        return results
        
    def _analyze_metric(self, metric: str) -> Optional[ABTestResult]:
        """Analyze specific metric for A/B test"""
        # Get model performance data
        metrics_a = self.performance_tracker.get_model_metrics("model_a")
        metrics_b = self.performance_tracker.get_model_metrics("model_b")
        
        if metric not in metrics_a or metric not in metrics_b:
            logger.warning(f"Metric {metric} not available for analysis")
            return None
            
        # Get raw values for statistical analysis
        model_a_data = [
            record for record in self.performance_tracker.performance_data
            if record['model_name'] == 'model_a' and record['actual'] is not None
        ]
        
        model_b_data = [
            record for record in self.performance_tracker.performance_data
            if record['model_name'] == 'model_b' and record['actual'] is not None
        ]
        
        if len(model_a_data) < self.config.minimum_sample_size or \
           len(model_b_data) < self.config.minimum_sample_size:
            logger.info(f"Insufficient sample size for {metric} analysis")
            return None
            
        # Extract metric values
        if metric == 'accuracy':
            values_a = [1.0 if r['correct'] else 0.0 for r in model_a_data]
            values_b = [1.0 if r['correct'] else 0.0 for r in model_b_data]
        elif metric in ['latency', 'average_latency']:
            values_a = [r['latency'] for r in model_a_data]
            values_b = [r['latency'] for r in model_b_data]
        else:
            # Use aggregated metrics
            values_a = [metrics_a[metric]] * len(model_a_data)
            values_b = [metrics_b[metric]] * len(model_b_data)
            
        # Perform statistical analysis
        test_result = self.analyzer.statistical_test(values_a, values_b)
        confidence_interval = self.analyzer.calculate_confidence_interval(values_a, values_b)
        effect_size = self.analyzer.calculate_effect_size(values_a, values_b)
        power = self.analyzer.calculate_power(values_a, values_b)
        
        # Calculate improvement
        model_a_value = np.mean(values_a)
        model_b_value = np.mean(values_b)
        improvement = model_b_value - model_a_value
        improvement_percentage = (improvement / model_a_value * 100) if model_a_value != 0 else 0
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            test_result['significant'], improvement_percentage, effect_size, power
        )
        
        return ABTestResult(
            test_name=self.config.name,
            model_a_name=self.config.model_a_path,
            model_b_name=self.config.model_b_path,
            metric=metric,
            model_a_value=model_a_value,
            model_b_value=model_b_value,
            improvement=improvement,
            improvement_percentage=improvement_percentage,
            statistical_significance=test_result['significant'],
            p_value=test_result['p_value'],
            confidence_interval=confidence_interval,
            sample_size_a=len(values_a),
            sample_size_b=len(values_b),
            power=power,
            effect_size=effect_size,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
        
    def _generate_recommendation(self, 
                               significant: bool, 
                               improvement_pct: float,
                               effect_size: float, 
                               power: float) -> str:
        """Generate actionable recommendation based on test results"""
        if not significant:
            if power < self.config.power_threshold:
                return "CONTINUE: Insufficient statistical power. Collect more data or increase effect size."
            else:
                return "STOP: No significant difference detected with adequate power. Keep current model."
                
        if improvement_pct > 0:
            if effect_size >= self.config.minimum_effect_size:
                return "DEPLOY: Model B shows significant improvement with meaningful effect size."
            else:
                return "CAUTION: Significant but small effect size. Consider business impact vs. complexity."
        else:
            return "ROLLBACK: Model B performs significantly worse. Revert to Model A."
            
    def should_conclude(self) -> bool:
        """Check if experiment should be concluded"""
        if not self.config.auto_conclude:
            return False
            
        # Time-based conclusion
        if (datetime.now() - self.start_time).days >= self.config.duration_days:
            return True
            
        # Sample size-based conclusion
        model_a_size = len([r for r in self.performance_tracker.performance_data 
                           if r['model_name'] == 'model_a' and r['actual'] is not None])
        model_b_size = len([r for r in self.performance_tracker.performance_data 
                           if r['model_name'] == 'model_b' and r['actual'] is not None])
        
        if (model_a_size >= self.config.minimum_sample_size and 
            model_b_size >= self.config.minimum_sample_size):
            
            # Check if we have significant results
            results = self.analyze_results()
            if any(result.statistical_significance for result in results):
                return True
                
        return False
        
    def conclude_experiment(self) -> Dict[str, Any]:
        """Conclude the experiment and generate final report"""
        self.status = "concluded"
        final_results = self.analyze_results()
        
        # Generate summary
        summary = {
            'experiment_name': self.config.name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_days': (datetime.now() - self.start_time).days,
            'total_predictions': len(self.performance_tracker.performance_data),
            'model_a_predictions': len([r for r in self.performance_tracker.performance_data 
                                       if r['model_name'] == 'model_a']),
            'model_b_predictions': len([r for r in self.performance_tracker.performance_data 
                                       if r['model_name'] == 'model_b']),
            'results': [asdict(result) for result in final_results],
            'overall_recommendation': self._get_overall_recommendation(final_results)
        }
        
        # Save results to MLflow
        self._log_to_mlflow(summary)
        
        return summary
        
    def _get_overall_recommendation(self, results: List[ABTestResult]) -> str:
        """Get overall recommendation based on all metric results"""
        if not results:
            return "INCONCLUSIVE: No analyzable results"
            
        deploy_count = sum(1 for r in results if "DEPLOY" in r.recommendation)
        total_metrics = len(results)
        
        if deploy_count == total_metrics:
            return "DEPLOY: All metrics show improvement"
        elif deploy_count > total_metrics / 2:
            return "DEPLOY WITH CAUTION: Most metrics show improvement"
        elif deploy_count > 0:
            return "MIXED RESULTS: Some metrics improved. Review individual metric analysis"
        else:
            return "DO NOT DEPLOY: No significant improvements detected"
            
    def _log_to_mlflow(self, summary: Dict[str, Any]):
        """Log experiment results to MLflow"""
        with mlflow.start_run(run_name=f"ab_test_{self.config.name}"):
            # Log experiment parameters
            mlflow.log_params({
                'experiment_name': self.config.name,
                'traffic_split': self.config.traffic_split,
                'duration_days': self.config.duration_days,
                'significance_level': self.config.significance_level,
                'minimum_sample_size': self.config.minimum_sample_size
            })
            
            # Log results
            for result in self.results:
                mlflow.log_metrics({
                    f"{result.metric}_model_a": result.model_a_value,
                    f"{result.metric}_model_b": result.model_b_value,
                    f"{result.metric}_improvement_pct": result.improvement_percentage,
                    f"{result.metric}_p_value": result.p_value,
                    f"{result.metric}_effect_size": result.effect_size,
                    f"{result.metric}_power": result.power
                })
                
            # Log overall metrics
            mlflow.log_metrics({
                'total_predictions': summary['total_predictions'],
                'model_a_predictions': summary['model_a_predictions'],
                'model_b_predictions': summary['model_b_predictions'],
                'experiment_duration_days': summary['duration_days']
            })
            
            # Log summary as artifact
            summary_path = "ab_test_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            mlflow.log_artifact(summary_path)
            
            mlflow.set_tag("experiment_type", "ab_test")
            mlflow.set_tag("overall_recommendation", summary['overall_recommendation'])

class ABTestVisualizer:
    """Generate visualizations for A/B test results"""
    
    def __init__(self, output_dir: str = "ab_test_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_metric_comparison(self, 
                              result: ABTestResult, 
                              save_path: Optional[str] = None) -> str:
        """Plot metric comparison between models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot comparison
        models = ['Model A', 'Model B']
        values = [result.model_a_value, result.model_b_value]
        colors = ['blue', 'green' if result.improvement > 0 else 'red']
        
        bars = ax1.bar(models, values, color=colors, alpha=0.7)
        ax1.set_ylabel(result.metric.title())
        ax1.set_title(f'{result.metric.title()} Comparison')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom')
                    
        # Statistical significance info
        significance_text = (
            f"Improvement: {result.improvement_percentage:.2f}%\n"
            f"P-value: {result.p_value:.4f}\n"
            f"Significant: {'Yes' if result.statistical_significance else 'No'}\n"
            f"Effect Size: {result.effect_size:.3f}\n"
            f"Power: {result.power:.3f}\n"
            f"CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]"
        )
        
        ax2.text(0.1, 0.7, significance_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax2.text(0.1, 0.3, f"Recommendation:\n{result.recommendation}", 
                transform=ax2.transAxes, fontsize=12, weight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        ax2.set_title('Statistical Analysis')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{result.metric}_comparison.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def plot_experiment_timeline(self, 
                               performance_tracker: ModelPerformanceTracker,
                               save_path: Optional[str] = None) -> str:
        """Plot experiment timeline and metrics over time"""
        df = pd.DataFrame(performance_tracker.performance_data)
        
        if df.empty:
            return None
            
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time-based metrics
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Calculate hourly metrics
        hourly_metrics = []
        for hour in df['hour'].unique():
            hour_data = df[df['hour'] == hour]
            
            for model in ['model_a', 'model_b']:
                model_data = hour_data[hour_data['model_name'] == model]
                if len(model_data) > 0:
                    accuracy = model_data['correct'].mean() if 'correct' in model_data.columns else np.nan
                    avg_latency = model_data['latency'].mean()
                    prediction_count = len(model_data)
                    
                    hourly_metrics.append({
                        'hour': hour,
                        'model': model,
                        'accuracy': accuracy,
                        'avg_latency': avg_latency,
                        'prediction_count': prediction_count
                    })
                    
        if not hourly_metrics:
            return None
            
        hourly_df = pd.DataFrame(hourly_metrics)
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Accuracy over time
        for model in ['model_a', 'model_b']:
            model_data = hourly_df[hourly_df['model'] == model]
            if not model_data.empty:
                axes[0].plot(model_data['hour'], model_data['accuracy'], 
                           marker='o', label=f'{model.replace("_", " ").title()}')
                           
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Latency over time
        for model in ['model_a', 'model_b']:
            model_data = hourly_df[hourly_df['model'] == model]
            if not model_data.empty:
                axes[1].plot(model_data['hour'], model_data['avg_latency'], 
                           marker='s', label=f'{model.replace("_", " ").title()}')
                           
        axes[1].set_ylabel('Average Latency (s)')
        axes[1].set_title('Model Latency Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Prediction count over time
        for model in ['model_a', 'model_b']:
            model_data = hourly_df[hourly_df['model'] == model]
            if not model_data.empty:
                axes[2].plot(model_data['hour'], model_data['prediction_count'], 
                           marker='^', label=f'{model.replace("_", " ").title()}')
                           
        axes[2].set_ylabel('Predictions per Hour')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Prediction Volume Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "experiment_timeline.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

def create_ab_test_config():
    """Create sample A/B test configuration"""
    config = {
        "experiment_name": "iris_classifier_v2_test",
        "model_a_path": "models/iris_classifier_v1.joblib",
        "model_b_path": "models/iris_classifier_v2.joblib",
        "traffic_split": 0.5,
        "duration_days": 7,
        "success_metrics": ["accuracy", "precision", "recall", "f1_score", "latency"],
        "minimum_sample_size": 100,
        "significance_level": 0.05,
        "power_threshold": 0.8,
        "minimum_effect_size": 0.2,
        "auto_conclude": True
    }
    
    with open("ab_test_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Created ab_test_config.json")

if __name__ == "__main__":
    # Create sample configuration
    create_ab_test_config()
    
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create two slightly different models
    model_a = RandomForestClassifier(n_estimators=50, random_state=42)
    model_b = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model_a.fit(X, y)
    model_b.fit(X, y)
    
    # Create experiment configuration
    config = ExperimentConfig(
        name="iris_test_experiment",
        model_a_path="model_a.joblib",
        model_b_path="model_b.joblib",
        traffic_split=0.5,
        duration_days=1,
        success_metrics=["accuracy"],
        minimum_sample_size=50,
        significance_level=0.05,
        power_threshold=0.8,
        minimum_effect_size=0.1,
        auto_conclude=True
    )
    
    # Run experiment
    experiment = ABTestExperiment(config)
    experiment.setup_models(model_a, model_b)
    
    # Simulate some predictions
    for i in range(100):
        features = {f'feature_{j}': X[i % len(X)][j] for j in range(X.shape[1])}
        model_name, prediction = experiment.route_prediction(features, user_id=f"user_{i}")
        
        # Simulate ground truth (with some delay)
        if i >= 10:  # Simulate labeling delay
            actual = y[(i-10) % len(y)]
            experiment.update_ground_truth(f"user_{i-10}", actual)
            
    # Analyze results
    results = experiment.analyze_results()
    for result in results:
        print(f"Metric: {result.metric}")
        print(f"Improvement: {result.improvement_percentage:.2f}%")
        print(f"Significant: {result.statistical_significance}")
        print(f"Recommendation: {result.recommendation}")
        print("---")