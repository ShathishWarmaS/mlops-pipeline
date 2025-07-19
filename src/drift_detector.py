#!/usr/bin/env python3
"""
Advanced Data Drift Detection System for MLOps
Comprehensive drift detection using multiple statistical methods and visualization
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
from scipy.spatial.distance import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: datetime
    overall_drift_score: float
    drift_detected: bool
    drift_method: str
    feature_scores: Dict[str, float]
    drift_threshold: float
    affected_features: List[str]
    severity: str  # low, medium, high, critical
    recommendations: List[str]
    statistical_tests: Dict[str, Dict[str, float]]
    visualizations: List[str]

class StatisticalDriftDetector:
    """Statistical drift detection using multiple methods"""
    
    def __init__(self, 
                 drift_threshold: float = 0.1,
                 significance_level: float = 0.05,
                 methods: List[str] = None):
        """
        Initialize drift detector
        
        Args:
            drift_threshold: Threshold for detecting drift
            significance_level: Statistical significance level
            methods: List of methods to use ['psi', 'ks', 'wasserstein', 'chi2']
        """
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.methods = methods or ['psi', 'ks', 'wasserstein', 'chi2']
        self.reference_data = None
        self.reference_stats = None
        
    def fit(self, reference_data: pd.DataFrame):
        """Fit the detector on reference data"""
        self.reference_data = reference_data.copy()
        self.reference_stats = self._calculate_reference_stats(reference_data)
        logger.info(f"Drift detector fitted on {len(reference_data)} reference samples")
        
    def _calculate_reference_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for reference data"""
        stats = {}
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                col_data = data[column].dropna()
                stats[column] = {
                    'type': 'numerical',
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'percentiles': {
                        'p5': col_data.quantile(0.05),
                        'p25': col_data.quantile(0.25),
                        'p75': col_data.quantile(0.75),
                        'p95': col_data.quantile(0.95)
                    },
                    'histogram': np.histogram(col_data, bins=20)
                }
            else:
                # Categorical data
                value_counts = data[column].value_counts(normalize=True)
                stats[column] = {
                    'type': 'categorical',
                    'categories': value_counts.index.tolist(),
                    'proportions': value_counts.values.tolist(),
                    'unique_count': data[column].nunique()
                }
                
        return stats
        
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Handle edge cases
            if len(reference) == 0 or len(current) == 0:
                return 1.0
                
            # Create bins based on reference data
            if reference.dtype in ['int64', 'float64']:
                # Numerical data
                _, bin_edges = np.histogram(reference, bins=bins)
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                
                ref_counts, _ = np.histogram(reference, bins=bin_edges)
                cur_counts, _ = np.histogram(current, bins=bin_edges)
            else:
                # Categorical data
                all_categories = set(reference.unique()) | set(current.unique())
                ref_counts = pd.Series(index=all_categories, dtype=int).fillna(0)
                cur_counts = pd.Series(index=all_categories, dtype=int).fillna(0)
                
                for cat in all_categories:
                    ref_counts[cat] = (reference == cat).sum()
                    cur_counts[cat] = (current == cat).sum()
                    
                ref_counts = ref_counts.values
                cur_counts = cur_counts.values
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Avoid division by zero and log of zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            cur_props = np.where(cur_props == 0, 0.0001, cur_props)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 1.0
            
    def _calculate_ks_statistic(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov test statistic and p-value"""
        try:
            if reference.dtype in ['int64', 'float64'] and current.dtype in ['int64', 'float64']:
                statistic, p_value = stats.ks_2samp(reference.dropna(), current.dropna())
                return statistic, p_value
            else:
                # For categorical data, use chi-square test
                return self._calculate_chi2_statistic(reference, current)
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return 1.0, 0.0
            
    def _calculate_chi2_statistic(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """Calculate Chi-square test statistic for categorical data"""
        try:
            # Get all unique categories
            all_categories = set(reference.unique()) | set(current.unique())
            
            # Create contingency table
            ref_counts = []
            cur_counts = []
            
            for cat in all_categories:
                ref_counts.append((reference == cat).sum())
                cur_counts.append((current == cat).sum())
                
            # Perform chi-square test
            observed = np.array([ref_counts, cur_counts])
            statistic, p_value, _, _ = stats.chi2_contingency(observed)
            
            return statistic, p_value
            
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return 1.0, 0.0
            
    def _calculate_wasserstein_distance(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Wasserstein distance for numerical data"""
        try:
            if reference.dtype in ['int64', 'float64'] and current.dtype in ['int64', 'float64']:
                return wasserstein_distance(reference.dropna(), current.dropna())
            else:
                # For categorical data, convert to numerical encoding
                all_categories = list(set(reference.unique()) | set(current.unique()))
                cat_to_num = {cat: i for i, cat in enumerate(all_categories)}
                
                ref_num = reference.map(cat_to_num).dropna()
                cur_num = current.map(cat_to_num).dropna()
                
                return wasserstein_distance(ref_num, cur_num)
                
        except Exception as e:
            logger.warning(f"Wasserstein distance calculation failed: {e}")
            return 1.0
            
    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """Detect drift in current data compared to reference"""
        if self.reference_data is None:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        feature_scores = {}
        statistical_tests = {}
        affected_features = []
        
        for feature in self.reference_data.columns:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in current data")
                continue
                
            ref_feature = self.reference_data[feature]
            cur_feature = current_data[feature]
            
            feature_results = {}
            
            # Calculate drift scores using different methods
            if 'psi' in self.methods:
                psi_score = self._calculate_psi(ref_feature, cur_feature)
                feature_results['psi'] = psi_score
                
            if 'ks' in self.methods:
                ks_stat, ks_p = self._calculate_ks_statistic(ref_feature, cur_feature)
                feature_results['ks_statistic'] = ks_stat
                feature_results['ks_p_value'] = ks_p
                
            if 'wasserstein' in self.methods:
                wasserstein_dist = self._calculate_wasserstein_distance(ref_feature, cur_feature)
                feature_results['wasserstein'] = wasserstein_dist
                
            if 'chi2' in self.methods and ref_feature.dtype not in ['int64', 'float64']:
                chi2_stat, chi2_p = self._calculate_chi2_statistic(ref_feature, cur_feature)
                feature_results['chi2_statistic'] = chi2_stat
                feature_results['chi2_p_value'] = chi2_p
                
            statistical_tests[feature] = feature_results
            
            # Determine primary drift score (using PSI as default)
            if 'psi' in feature_results:
                drift_score = feature_results['psi']
            elif 'ks_statistic' in feature_results:
                drift_score = feature_results['ks_statistic']
            elif 'wasserstein' in feature_results:
                # Normalize Wasserstein distance
                ref_std = ref_feature.std() if ref_feature.dtype in ['int64', 'float64'] else 1.0
                drift_score = feature_results['wasserstein'] / max(ref_std, 0.01)
            else:
                drift_score = 0.0
                
            feature_scores[feature] = drift_score
            
            # Check if feature shows significant drift
            if drift_score > self.drift_threshold:
                affected_features.append(feature)
                
        # Calculate overall drift score
        if feature_scores:
            overall_drift_score = np.mean(list(feature_scores.values()))
        else:
            overall_drift_score = 0.0
            
        # Determine drift detection
        drift_detected = overall_drift_score > self.drift_threshold
        
        # Determine severity
        if overall_drift_score < self.drift_threshold:
            severity = "low"
        elif overall_drift_score < self.drift_threshold * 2:
            severity = "medium"
        elif overall_drift_score < self.drift_threshold * 3:
            severity = "high"
        else:
            severity = "critical"
            
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_drift_score, affected_features, severity
        )
        
        return DriftReport(
            timestamp=datetime.now(),
            overall_drift_score=overall_drift_score,
            drift_detected=drift_detected,
            drift_method="multi_method",
            feature_scores=feature_scores,
            drift_threshold=self.drift_threshold,
            affected_features=affected_features,
            severity=severity,
            recommendations=recommendations,
            statistical_tests=statistical_tests,
            visualizations=[]
        )
        
    def _generate_recommendations(self, 
                                drift_score: float, 
                                affected_features: List[str], 
                                severity: str) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        if severity == "low":
            recommendations.append("Continue monitoring - drift levels are within acceptable range")
            
        elif severity == "medium":
            recommendations.extend([
                "Increase monitoring frequency for affected features",
                "Consider collecting more recent training data",
                "Review data preprocessing pipeline for consistency"
            ])
            
        elif severity == "high":
            recommendations.extend([
                "Immediate investigation required for affected features",
                "Consider model retraining with recent data",
                "Implement online learning or model adaptation",
                "Review data collection and preprocessing procedures"
            ])
            
        elif severity == "critical":
            recommendations.extend([
                "URGENT: Model performance likely degraded significantly",
                "Immediate model retraining recommended",
                "Consider rolling back to previous model version",
                "Investigate root cause of data distribution changes",
                "Implement emergency monitoring and alerting"
            ])
            
        if affected_features:
            recommendations.append(
                f"Focus investigation on features: {', '.join(affected_features[:5])}"
            )
            
        return recommendations

class VisualizationGenerator:
    """Generate drift detection visualizations"""
    
    def __init__(self, output_dir: str = "drift_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_feature_distributions(self, 
                                 reference_data: pd.DataFrame,
                                 current_data: pd.DataFrame,
                                 feature: str,
                                 save_path: Optional[str] = None) -> str:
        """Plot distribution comparison for a single feature"""
        plt.figure(figsize=(12, 6))
        
        if reference_data[feature].dtype in ['int64', 'float64']:
            # Numerical feature
            plt.subplot(1, 2, 1)
            plt.hist(reference_data[feature].dropna(), bins=30, alpha=0.7, 
                    label='Reference', density=True, color='blue')
            plt.hist(current_data[feature].dropna(), bins=30, alpha=0.7, 
                    label='Current', density=True, color='red')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.title(f'{feature} Distribution Comparison')
            plt.legend()
            
            # Box plot
            plt.subplot(1, 2, 2)
            data_to_plot = [reference_data[feature].dropna(), current_data[feature].dropna()]
            plt.boxplot(data_to_plot, labels=['Reference', 'Current'])
            plt.ylabel(feature)
            plt.title(f'{feature} Box Plot Comparison')
            
        else:
            # Categorical feature
            ref_counts = reference_data[feature].value_counts(normalize=True)
            cur_counts = current_data[feature].value_counts(normalize=True)
            
            # Align indices
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
            cur_aligned = cur_counts.reindex(all_categories, fill_value=0)
            
            x = np.arange(len(all_categories))
            width = 0.35
            
            plt.bar(x - width/2, ref_aligned.values, width, label='Reference', alpha=0.7)
            plt.bar(x + width/2, cur_aligned.values, width, label='Current', alpha=0.7)
            
            plt.xlabel('Categories')
            plt.ylabel('Proportion')
            plt.title(f'{feature} Category Distribution Comparison')
            plt.xticks(x, all_categories, rotation=45)
            plt.legend()
            
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"{feature}_distribution.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def plot_drift_summary(self, 
                          drift_report: DriftReport,
                          save_path: Optional[str] = None) -> str:
        """Plot drift detection summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature drift scores
        features = list(drift_report.feature_scores.keys())
        scores = list(drift_report.feature_scores.values())
        
        colors = ['red' if score > drift_report.drift_threshold else 'green' for score in scores]
        
        ax1.barh(features, scores, color=colors, alpha=0.7)
        ax1.axvline(x=drift_report.drift_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {drift_report.drift_threshold}')
        ax1.set_xlabel('Drift Score')
        ax1.set_title('Feature Drift Scores')
        ax1.legend()
        
        # Overall drift gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        ax2.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2)
        ax2.fill_between(r * np.cos(theta[:25]), r * np.sin(theta[:25]), alpha=0.3, color='green', label='Low')
        ax2.fill_between(r * np.cos(theta[25:50]), r * np.sin(theta[25:50]), alpha=0.3, color='yellow', label='Medium')
        ax2.fill_between(r * np.cos(theta[50:75]), r * np.sin(theta[50:75]), alpha=0.3, color='orange', label='High')
        ax2.fill_between(r * np.cos(theta[75:]), r * np.sin(theta[75:]), alpha=0.3, color='red', label='Critical')
        
        # Needle position
        score_angle = np.pi * (1 - min(drift_report.overall_drift_score, 1.0))
        ax2.arrow(0, 0, 0.8 * np.cos(score_angle), 0.8 * np.sin(score_angle), 
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_aspect('equal')
        ax2.set_title(f'Overall Drift Score: {drift_report.overall_drift_score:.3f}')
        ax2.legend(loc='upper right')
        ax2.axis('off')
        
        # Timeline (if multiple reports available)
        ax3.text(0.5, 0.5, f'Drift Detected: {drift_report.drift_detected}\n'
                           f'Severity: {drift_report.severity.title()}\n'
                           f'Affected Features: {len(drift_report.affected_features)}\n'
                           f'Timestamp: {drift_report.timestamp.strftime("%Y-%m-%d %H:%M")}',
                ha='center', va='center', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=12)
        ax3.set_title('Drift Detection Summary')
        ax3.axis('off')
        
        # Recommendations
        rec_text = '\n'.join([f'• {rec}' for rec in drift_report.recommendations[:5]])
        ax4.text(0.05, 0.95, 'Recommendations:\n\n' + rec_text,
                ha='left', va='top', transform=ax4.transAxes,
                fontsize=10, wrap=True)
        ax4.set_title('Action Items')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = drift_report.timestamp.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"drift_summary_{timestamp}.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def plot_pca_drift(self, 
                      reference_data: pd.DataFrame,
                      current_data: pd.DataFrame,
                      save_path: Optional[str] = None) -> str:
        """Plot PCA-based drift visualization"""
        # Select only numerical features
        numerical_features = reference_data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_features) < 2:
            logger.warning("Not enough numerical features for PCA drift plot")
            return None
            
        # Combine data for PCA
        ref_num = reference_data[numerical_features].fillna(0)
        cur_num = current_data[numerical_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(ref_num)
        cur_scaled = scaler.transform(cur_num)
        
        # Apply PCA
        pca = PCA(n_components=2)
        ref_pca = pca.fit_transform(ref_scaled)
        cur_pca = pca.transform(cur_scaled)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.6, label='Reference', s=20)
        plt.scatter(cur_pca[:, 0], cur_pca[:, 1], alpha=0.6, label='Current', s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA-based Data Drift Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "pca_drift.png")
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

class DriftDetectionPipeline:
    """Complete drift detection pipeline with reporting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.detector = StatisticalDriftDetector(
            drift_threshold=self.config['drift_threshold'],
            significance_level=self.config['significance_level'],
            methods=self.config['methods']
        )
        self.visualizer = VisualizationGenerator(self.config['output_dir'])
        self.drift_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for drift detection"""
        return {
            'drift_threshold': 0.1,
            'significance_level': 0.05,
            'methods': ['psi', 'ks', 'wasserstein'],
            'output_dir': 'drift_analysis',
            'generate_plots': True,
            'save_reports': True
        }
        
    def fit_reference(self, reference_data: pd.DataFrame):
        """Fit pipeline on reference data"""
        self.detector.fit(reference_data)
        self.reference_data = reference_data
        logger.info("Drift detection pipeline fitted on reference data")
        
    def detect_and_analyze(self, current_data: pd.DataFrame) -> DriftReport:
        """Run complete drift detection and analysis"""
        # Detect drift
        drift_report = self.detector.detect_drift(current_data)
        
        # Generate visualizations if configured
        if self.config['generate_plots']:
            plots = []
            
            # Overall summary plot
            summary_plot = self.visualizer.plot_drift_summary(drift_report)
            plots.append(summary_plot)
            
            # PCA drift plot
            if hasattr(self, 'reference_data'):
                pca_plot = self.visualizer.plot_pca_drift(self.reference_data, current_data)
                if pca_plot:
                    plots.append(pca_plot)
                    
            # Individual feature plots for drifted features
            for feature in drift_report.affected_features[:5]:  # Limit to top 5
                feature_plot = self.visualizer.plot_feature_distributions(
                    self.reference_data, current_data, feature
                )
                plots.append(feature_plot)
                
            drift_report.visualizations = plots
            
        # Save report if configured
        if self.config['save_reports']:
            self._save_report(drift_report)
            
        # Add to history
        self.drift_history.append(drift_report)
        
        return drift_report
        
    def _save_report(self, drift_report: DriftReport):
        """Save drift report to JSON file"""
        timestamp = drift_report.timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.config['output_dir'], f"drift_report_{timestamp}.json")
        
        # Convert report to JSON-serializable format
        report_dict = asdict(drift_report)
        report_dict['timestamp'] = drift_report.timestamp.isoformat()
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
            
        logger.info(f"Drift report saved to {report_path}")
        
    def get_drift_trends(self) -> Dict[str, Any]:
        """Analyze drift trends over time"""
        if len(self.drift_history) < 2:
            return {"message": "Not enough data for trend analysis"}
            
        # Overall drift trend
        timestamps = [report.timestamp for report in self.drift_history]
        scores = [report.overall_drift_score for report in self.drift_history]
        
        # Feature-wise trends
        feature_trends = {}
        for report in self.drift_history:
            for feature, score in report.feature_scores.items():
                if feature not in feature_trends:
                    feature_trends[feature] = []
                feature_trends[feature].append(score)
                
        return {
            'timestamps': timestamps,
            'overall_scores': scores,
            'feature_trends': feature_trends,
            'total_reports': len(self.drift_history),
            'latest_severity': self.drift_history[-1].severity
        }

def create_drift_config():
    """Create sample drift detection configuration"""
    config = {
        "drift_threshold": 0.1,
        "significance_level": 0.05,
        "methods": ["psi", "ks", "wasserstein", "chi2"],
        "output_dir": "drift_analysis",
        "generate_plots": True,
        "save_reports": True,
        "monitoring": {
            "check_interval_hours": 6,
            "alert_threshold": 0.15,
            "email_alerts": True,
            "slack_alerts": True
        }
    }
    
    with open("drift_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Created drift_config.json")

if __name__ == "__main__":
    # Create sample configuration
    create_drift_config()
    
    # Example usage
    from sklearn.datasets import load_iris
    
    # Load sample data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Split into reference and current (with some artificial drift)
    reference_data = data.iloc[:100]
    current_data = data.iloc[50:].copy()
    
    # Add some artificial drift
    current_data.iloc[:, 0] += np.random.normal(0.5, 0.2, len(current_data))
    
    # Initialize pipeline
    pipeline = DriftDetectionPipeline()
    pipeline.fit_reference(reference_data)
    
    # Detect drift
    report = pipeline.detect_and_analyze(current_data)
    
    print(f"Drift detected: {report.drift_detected}")
    print(f"Overall score: {report.overall_drift_score:.3f}")
    print(f"Severity: {report.severity}")
    print(f"Affected features: {report.affected_features}")