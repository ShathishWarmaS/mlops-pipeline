#!/usr/bin/env python3
"""
Model Explainability and Interpretability Tools for MLOps
Comprehensive model interpretation using SHAP, LIME, and custom techniques
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML interpretability libraries
import shap
import lime
import lime.lime_tabular
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Result from model explanation analysis"""
    model_name: str
    model_version: str
    explanation_type: str  # global, local, feature_importance
    method: str  # shap, lime, permutation, etc.
    timestamp: datetime
    sample_id: Optional[str] = None
    prediction: Optional[Any] = None
    explanation_values: Dict[str, float] = None
    feature_names: List[str] = None
    visualization_paths: List[str] = None
    interpretation: str = ""
    confidence_score: float = 0.0

@dataclass
class GlobalExplanation:
    """Global model explanation"""
    model_name: str
    feature_importance: Dict[str, float]
    feature_interactions: Dict[str, Dict[str, float]]
    model_complexity: Dict[str, Any]
    interpretation_summary: str
    timestamp: datetime

class SHAPExplainer:
    """SHAP-based model explanations"""
    
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        self.explainer = None
        self._initialize_explainer()
        
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            if hasattr(self.model, 'tree_'):
                # Tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer for tree-based model")
            elif hasattr(self.model, 'coef_'):
                # Linear models
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                logger.info("Initialized LinearExplainer for linear model")
            else:
                # General case - use KernelExplainer (slower but works for any model)
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                logger.info("Initialized KernelExplainer for general model")
        except Exception as e:
            logger.warning(f"Failed to initialize specific explainer, using KernelExplainer: {e}")
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            
    def explain_instance(self, X_instance: Union[pd.DataFrame, np.ndarray], 
                        sample_id: str = None) -> ExplanationResult:
        """Explain a single prediction"""
        if isinstance(X_instance, pd.DataFrame):
            X_array = X_instance.values
        else:
            X_array = X_instance.reshape(1, -1) if X_instance.ndim == 1 else X_array
            
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_array)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # For multi-class, use the predicted class
            prediction = self.model.predict(X_array)[0]
            class_idx = int(prediction) if hasattr(prediction, '__int__') else 0
            explanation_values = dict(zip(self.feature_names, shap_values[class_idx][0]))
        else:
            explanation_values = dict(zip(self.feature_names, shap_values[0]))
            prediction = self.model.predict(X_array)[0]
            
        # Calculate confidence score (based on magnitude of SHAP values)
        confidence_score = np.abs(list(explanation_values.values())).mean()
        
        # Generate interpretation
        top_features = sorted(explanation_values.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:3]
        interpretation = f"Top contributing features: {', '.join([f[0] for f in top_features])}"
        
        return ExplanationResult(
            model_name=getattr(self.model, '__class__').__name__,
            model_version="current",
            explanation_type="local",
            method="shap",
            timestamp=datetime.now(),
            sample_id=sample_id,
            prediction=prediction,
            explanation_values=explanation_values,
            feature_names=self.feature_names,
            interpretation=interpretation,
            confidence_score=confidence_score
        )
        
    def explain_global(self, X_sample: pd.DataFrame = None) -> GlobalExplanation:
        """Generate global model explanation"""
        if X_sample is None:
            X_sample = self.X_train.sample(min(1000, len(self.X_train)))
            
        # Get SHAP values for sample
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Average across classes
            avg_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            feature_importance = dict(zip(self.feature_names, np.mean(avg_shap_values, axis=0)))
        else:
            feature_importance = dict(zip(self.feature_names, np.mean(np.abs(shap_values), axis=0)))
            
        # Calculate feature interactions (simplified)
        feature_interactions = {}
        for i, feat1 in enumerate(self.feature_names):
            feature_interactions[feat1] = {}
            for j, feat2 in enumerate(self.feature_names):
                if i != j:
                    # Simplified interaction calculation
                    correlation = X_sample.iloc[:, i].corr(X_sample.iloc[:, j])
                    importance_product = feature_importance[feat1] * feature_importance[feat2]
                    feature_interactions[feat1][feat2] = correlation * importance_product
                    
        # Model complexity metrics
        model_complexity = {
            'feature_count': len(self.feature_names),
            'total_importance': sum(feature_importance.values()),
            'importance_concentration': max(feature_importance.values()) / sum(feature_importance.values())
        }
        
        # Generate interpretation summary
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        interpretation = f"Model primarily relies on: {', '.join([f[0] for f in top_features])}"
        
        return GlobalExplanation(
            model_name=getattr(self.model, '__class__').__name__,
            feature_importance=feature_importance,
            feature_interactions=feature_interactions,
            model_complexity=model_complexity,
            interpretation_summary=interpretation,
            timestamp=datetime.now()
        )
        
    def plot_explanation(self, explanation_result: ExplanationResult, 
                        save_path: str = None) -> str:
        """Create visualization for SHAP explanation"""
        if save_path is None:
            save_path = f"shap_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        
        # Sort features by absolute importance
        features = list(explanation_result.explanation_values.keys())
        values = list(explanation_result.explanation_values.values())
        
        sorted_items = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
        sorted_features, sorted_values = zip(*sorted_items)
        
        # Create horizontal bar plot
        colors = ['red' if v < 0 else 'blue' for v in sorted_values]
        plt.barh(range(len(sorted_features)), sorted_values, color=colors, alpha=0.7)
        
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('SHAP Value (impact on model output)')
        plt.title(f'SHAP Feature Importance for Prediction: {explanation_result.prediction}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(sorted_values):
            plt.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', 
                    va='center', ha='left' if v >= 0 else 'right')
                    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

class LIMEExplainer:
    """LIME-based model explanations"""
    
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, mode: str = 'classification'):
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        self.mode = mode
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )
        
    def explain_instance(self, X_instance: Union[pd.DataFrame, np.ndarray],
                        sample_id: str = None, num_features: int = 10) -> ExplanationResult:
        """Explain a single prediction using LIME"""
        if isinstance(X_instance, pd.DataFrame):
            instance_array = X_instance.values.flatten()
        else:
            instance_array = X_instance.flatten() if X_instance.ndim > 1 else X_instance
            
        # Get LIME explanation
        if self.mode == 'classification':
            explanation = self.explainer.explain_instance(
                instance_array, 
                self.model.predict_proba,
                num_features=num_features
            )
        else:
            explanation = self.explainer.explain_instance(
                instance_array,
                self.model.predict,
                num_features=num_features
            )
            
        # Extract explanation values
        explanation_values = dict(explanation.as_list())
        
        # Map back to original feature names if needed
        mapped_values = {}
        for feat_desc, value in explanation_values.items():
            # LIME returns feature descriptions, map back to feature names
            for fname in self.feature_names:
                if fname in feat_desc:
                    mapped_values[fname] = value
                    break
            else:
                mapped_values[feat_desc] = value
                
        # Get prediction
        prediction = self.model.predict(instance_array.reshape(1, -1))[0]
        
        # Calculate confidence score
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(instance_array.reshape(1, -1))[0]
            confidence_score = max(proba)
        else:
            confidence_score = 0.8  # Default for regression
            
        # Generate interpretation
        top_features = sorted(mapped_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        interpretation = f"LIME explanation - key factors: {', '.join([f[0] for f in top_features])}"
        
        return ExplanationResult(
            model_name=getattr(self.model, '__class__').__name__,
            model_version="current",
            explanation_type="local",
            method="lime",
            timestamp=datetime.now(),
            sample_id=sample_id,
            prediction=prediction,
            explanation_values=mapped_values,
            feature_names=self.feature_names,
            interpretation=interpretation,
            confidence_score=confidence_score
        )

class PermutationExplainer:
    """Permutation importance-based explanations"""
    
    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = X_test.columns.tolist()
        
    def explain_global(self, scoring: str = 'accuracy', n_repeats: int = 10) -> GlobalExplanation:
        """Generate global explanation using permutation importance"""
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, self.X_test, self.y_test,
            scoring=scoring, n_repeats=n_repeats, random_state=42
        )
        
        # Extract feature importance
        feature_importance = dict(zip(self.feature_names, perm_importance.importances_mean))
        
        # Calculate feature interactions (correlation-based)
        feature_interactions = {}
        for i, feat1 in enumerate(self.feature_names):
            feature_interactions[feat1] = {}
            for j, feat2 in enumerate(self.feature_names):
                if i != j:
                    correlation = self.X_test.iloc[:, i].corr(self.X_test.iloc[:, j])
                    feature_interactions[feat1][feat2] = correlation
                    
        # Model complexity
        model_complexity = {
            'feature_count': len(self.feature_names),
            'importance_std': perm_importance.importances_std.tolist(),
            'total_importance': sum(feature_importance.values())
        }
        
        # Interpretation
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        interpretation = f"Permutation importance analysis shows {', '.join([f[0] for f in top_features])} as most important"
        
        return GlobalExplanation(
            model_name=getattr(self.model, '__class__').__name__,
            feature_importance=feature_importance,
            feature_interactions=feature_interactions,
            model_complexity=model_complexity,
            interpretation_summary=interpretation,
            timestamp=datetime.now()
        )

class ModelExplainabilityPipeline:
    """Comprehensive model explainability pipeline"""
    
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, 
                 X_test: pd.DataFrame = None, y_test: pd.Series = None,
                 output_dir: str = "explainability_results"):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.y_test = y_test
        self.output_dir = output_dir
        self.feature_names = X_train.columns.tolist()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, X_train)
        self.lime_explainer = LIMEExplainer(model, X_train)
        if y_test is not None:
            self.perm_explainer = PermutationExplainer(model, self.X_test, y_test)
        else:
            self.perm_explainer = None
            
        # Results storage
        self.explanations = []
        self.global_explanations = {}
        
    def explain_predictions(self, X_samples: pd.DataFrame, 
                          sample_ids: List[str] = None,
                          methods: List[str] = ['shap', 'lime']) -> List[ExplanationResult]:
        """Explain multiple predictions using specified methods"""
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(X_samples))]
            
        results = []
        
        for i, (idx, row) in enumerate(X_samples.iterrows()):
            sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
            instance = row.values.reshape(1, -1)
            
            for method in methods:
                try:
                    if method == 'shap':
                        result = self.shap_explainer.explain_instance(instance, sample_id)
                    elif method == 'lime':
                        result = self.lime_explainer.explain_instance(instance, sample_id)
                    else:
                        logger.warning(f"Unknown explanation method: {method}")
                        continue
                        
                    results.append(result)
                    self.explanations.append(result)
                    
                    # Create visualization
                    if method == 'shap':
                        viz_path = os.path.join(self.output_dir, 
                                              f"{method}_{sample_id}_explanation.png")
                        self.shap_explainer.plot_explanation(result, viz_path)
                        result.visualization_paths = [viz_path]
                        
                except Exception as e:
                    logger.error(f"Failed to explain {sample_id} with {method}: {e}")
                    
        return results
        
    def generate_global_explanations(self, methods: List[str] = ['shap', 'permutation']) -> Dict[str, GlobalExplanation]:
        """Generate global model explanations"""
        global_explanations = {}
        
        for method in methods:
            try:
                if method == 'shap':
                    explanation = self.shap_explainer.explain_global()
                elif method == 'permutation' and self.perm_explainer:
                    explanation = self.perm_explainer.explain_global()
                else:
                    logger.warning(f"Skipping method {method} - explainer not available")
                    continue
                    
                global_explanations[method] = explanation
                self.global_explanations[method] = explanation
                
                # Create visualizations
                self._create_global_visualizations(explanation, method)
                
            except Exception as e:
                logger.error(f"Failed to generate global explanation with {method}: {e}")
                
        return global_explanations
        
    def _create_global_visualizations(self, explanation: GlobalExplanation, method: str):
        """Create visualizations for global explanations"""
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        
        features = list(explanation.feature_importance.keys())
        importances = list(explanation.feature_importance.values())
        
        # Sort by importance
        sorted_items = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        sorted_features, sorted_importances = zip(*sorted_items)
        
        plt.subplot(2, 2, 1)
        plt.barh(range(len(sorted_features)), sorted_importances, alpha=0.7)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Global Feature Importance ({method.upper()})')
        
        # Feature interaction heatmap
        plt.subplot(2, 2, 2)
        if explanation.feature_interactions:
            interaction_df = pd.DataFrame(explanation.feature_interactions)
            sns.heatmap(interaction_df.fillna(0), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Interactions')
            
        # Importance distribution
        plt.subplot(2, 2, 3)
        plt.hist(importances, bins=min(10, len(importances)), alpha=0.7)
        plt.xlabel('Importance Value')
        plt.ylabel('Frequency')
        plt.title('Importance Distribution')
        
        # Model complexity metrics
        plt.subplot(2, 2, 4)
        complexity_metrics = explanation.model_complexity
        metrics_names = list(complexity_metrics.keys())
        metrics_values = list(complexity_metrics.values())
        
        plt.bar(metrics_names, metrics_values, alpha=0.7)
        plt.ylabel('Value')
        plt.title('Model Complexity Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, f"global_explanation_{method}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_model_report(self) -> Dict[str, Any]:
        """Generate comprehensive model explainability report"""
        report = {
            'model_info': {
                'model_type': getattr(self.model, '__class__').__name__,
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            },
            'global_explanations': {},
            'local_explanations_summary': {},
            'interpretation': {},
            'recommendations': []
        }
        
        # Add global explanations
        for method, explanation in self.global_explanations.items():
            report['global_explanations'][method] = {
                'feature_importance': explanation.feature_importance,
                'model_complexity': explanation.model_complexity,
                'interpretation': explanation.interpretation_summary
            }
            
        # Summarize local explanations
        if self.explanations:
            methods_used = set(exp.method for exp in self.explanations)
            for method in methods_used:
                method_explanations = [exp for exp in self.explanations if exp.method == method]
                
                # Calculate average feature importance across samples
                all_features = set()
                for exp in method_explanations:
                    all_features.update(exp.explanation_values.keys())
                    
                avg_importance = {}
                for feature in all_features:
                    values = [exp.explanation_values.get(feature, 0) for exp in method_explanations]
                    avg_importance[feature] = np.mean(values)
                    
                report['local_explanations_summary'][method] = {
                    'sample_count': len(method_explanations),
                    'average_feature_importance': avg_importance,
                    'average_confidence': np.mean([exp.confidence_score for exp in method_explanations])
                }
                
        # Generate interpretations and recommendations
        report['interpretation'] = self._generate_model_interpretation()
        report['recommendations'] = self._generate_recommendations()
        
        # Save report
        report_path = os.path.join(self.output_dir, "explainability_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
        
    def _generate_model_interpretation(self) -> Dict[str, str]:
        """Generate high-level model interpretation"""
        interpretation = {}
        
        # Overall model behavior
        if 'shap' in self.global_explanations:
            shap_exp = self.global_explanations['shap']
            top_features = sorted(shap_exp.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            interpretation['primary_drivers'] = f"Model decisions are primarily driven by: {', '.join([f[0] for f in top_features])}"
            
            # Complexity assessment
            complexity = shap_exp.model_complexity
            if complexity['importance_concentration'] > 0.5:
                interpretation['complexity'] = "Model shows high concentration on few features (low complexity)"
            else:
                interpretation['complexity'] = "Model distributes importance across many features (high complexity)"
                
        # Consistency analysis
        if len(self.explanations) > 1:
            methods = set(exp.method for exp in self.explanations)
            if len(methods) > 1:
                interpretation['consistency'] = "Multiple explanation methods available for cross-validation"
            else:
                interpretation['consistency'] = "Single explanation method used"
                
        return interpretation
        
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Feature-based recommendations
        if 'shap' in self.global_explanations:
            feature_importance = self.global_explanations['shap'].feature_importance
            top_feature = max(feature_importance.items(), key=lambda x: x[1])
            
            if top_feature[1] > 0.5 * sum(feature_importance.values()):
                recommendations.append(
                    f"Model heavily relies on '{top_feature[0]}' - consider feature engineering or validation"
                )
                
            # Low importance features
            low_importance = [f for f, imp in feature_importance.items() if imp < 0.01]
            if low_importance:
                recommendations.append(
                    f"Consider removing low-importance features: {', '.join(low_importance[:3])}"
                )
                
        # Model complexity recommendations
        if self.global_explanations:
            recommendations.append("Regular explainability analysis recommended for model monitoring")
            
        # Data quality recommendations
        recommendations.append("Validate feature quality and consistency in production data")
        
        return recommendations

def create_explainability_demo():
    """Create demonstration of explainability tools"""
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create explainability pipeline
    pipeline = ModelExplainabilityPipeline(model, X_train, X_test, y_test)
    
    # Generate explanations
    print("Generating local explanations...")
    sample_explanations = pipeline.explain_predictions(X_test.head(5))
    
    print("Generating global explanations...")
    global_explanations = pipeline.generate_global_explanations()
    
    print("Creating comprehensive report...")
    report = pipeline.generate_model_report()
    
    print(f"Explainability analysis completed!")
    print(f"Results saved to: {pipeline.output_dir}")
    print(f"Report summary:")
    print(f"- Local explanations: {len(sample_explanations)}")
    print(f"- Global explanations: {len(global_explanations)}")
    print(f"- Primary drivers: {report['interpretation'].get('primary_drivers', 'N/A')}")
    
    return pipeline, report

if __name__ == "__main__":
    # Run demonstration
    pipeline, report = create_explainability_demo()
    
    print("\nModel Explainability Demo completed successfully!")
    print("Check the 'explainability_results' directory for detailed outputs.")