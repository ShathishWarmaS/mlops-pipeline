"""ML model training and evaluation with MLflow tracking."""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import MLOpsConfig
from mlflow_manager import MLflowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisClassifier:
    """Iris classification model with enhanced MLflow tracking."""
    
    def __init__(self, config: MLOpsConfig, environment: str = "development"):
        self.config = config
        self.environment = environment
        self.mlflow_manager = MLflowManager(config)
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=config.random_state
        )
        self.is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray,
              run_name: str = None) -> Dict[str, Any]:
        """Train the model with enhanced MLflow tracking."""
        
        # Start enhanced MLflow run
        run_tags = {
            "model_version": "1.0.0",
            "training_data": "iris_dataset",
            "hyperparameters": "default_rf"
        }
        
        with self.mlflow_manager.start_run(
            environment=self.environment,
            run_name=run_name,
            tags=run_tags
        ):
            logger.info(f"Starting model training in {self.environment} environment...")
            
            # Log comprehensive data information
            self.mlflow_manager.log_data_info(X_train, X_test, y_train, y_test)
            
            # Train the model
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            y_pred_proba_test = self.model.predict_proba(X_test)
            
            # Calculate comprehensive metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Cross-validation scores
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            # ROC AUC for multiclass
            try:
                test_auc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='weighted')
            except:
                test_auc = 0.0
            
            # Compile all metrics
            metrics = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1,
                "test_auc": test_auc,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
                "cv_min_accuracy": cv_scores.min(),
                "cv_max_accuracy": cv_scores.max()
            }
            
            # Log all metrics using enhanced manager
            self.mlflow_manager.log_training_metrics(metrics, self.environment)
            
            # Log model with enhanced details
            model_uri = self.mlflow_manager.log_model_details(
                self.model, 
                self.config.model_name, 
                self.environment
            )
            
            # Create and log enhanced visualizations
            self._log_enhanced_visualizations(
                X_train, X_test, y_train, y_test, 
                y_pred_test, y_pred_proba_test, cv_scores
            )
            
            # Log classification report with detailed breakdown
            self._log_detailed_classification_report(y_test, y_pred_test)
            
            # Log model interpretation
            self._log_model_interpretation()
            
            logger.info(f"Model training completed in {self.environment}. Test accuracy: {test_accuracy:.4f}")
            
            return {
                **metrics,
                "run_id": mlflow.active_run().info.run_id,
                "model_uri": model_uri,
                "environment": self.environment
            }
    
    def _log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Create and log confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    yticklabels=['Setosa', 'Versicolor', 'Virginica'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
    
    def _log_enhanced_visualizations(self, X_train, X_test, y_train, y_test, 
                                   y_pred_test, y_pred_proba_test, cv_scores):
        """Create and log comprehensive visualizations."""
        
        # 1. Enhanced Confusion Matrix
        self._log_enhanced_confusion_matrix(y_test, y_pred_test)
        
        # 2. Feature Importance with enhanced details
        self._log_enhanced_feature_importance()
        
        # 3. Cross-validation scores plot
        self._log_cv_scores_plot(cv_scores)
        
        # 4. ROC Curves for each class
        self._log_roc_curves(y_test, y_pred_proba_test)
        
        # 5. Prediction confidence distribution
        self._log_prediction_confidence(y_pred_proba_test)
        
        # 6. Data distribution plots
        self._log_data_distribution(X_train, y_train)
    
    def _log_enhanced_confusion_matrix(self, y_true, y_pred):
        """Create enhanced confusion matrix with percentages."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    yticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig("enhanced_confusion_matrix.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("enhanced_confusion_matrix.png")
        plt.close()
    
    def _log_enhanced_feature_importance(self):
        """Enhanced feature importance with statistical details."""
        if not self.is_fitted:
            return
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        importances = self.model.feature_importances_
        
        # Get feature importance standard deviation from trees
        std_importances = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        # Create enhanced plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot with error bars
        indices = np.argsort(importances)[::-1]
        ax1.bar(range(len(importances)), importances[indices], 
                yerr=std_importances[indices], capsize=5)
        ax1.set_title('Feature Importance with Standard Deviation')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Importance')
        ax1.set_xticks(range(len(importances)))
        ax1.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        # Pie chart
        ax2.pie(importances, labels=feature_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Importance Distribution')
        
        plt.tight_layout()
        plt.savefig("enhanced_feature_importance.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("enhanced_feature_importance.png")
        plt.close()
        
        # Log detailed metrics
        for name, importance, std in zip(feature_names, importances, std_importances):
            mlflow.log_metric(f"feature_importance_{name}", importance)
            mlflow.log_metric(f"feature_importance_std_{name}", std)
    
    def _log_cv_scores_plot(self, cv_scores):
        """Plot cross-validation scores."""
        plt.figure(figsize=(10, 6))
        
        # Box plot and individual scores
        plt.subplot(1, 2, 1)
        plt.boxplot(cv_scores, labels=['CV Scores'])
        plt.title('Cross-Validation Score Distribution')
        plt.ylabel('Accuracy')
        
        # Line plot of CV scores
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
        plt.title('Cross-Validation Scores by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("cv_scores.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("cv_scores.png")
        plt.close()
    
    def _log_roc_curves(self, y_true, y_pred_proba):
        """Plot ROC curves for multiclass classification."""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = y_bin.shape[1]
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'blue', 'green']
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
            
            # Log AUC for each class
            mlflow.log_metric(f"auc_{class_names[i].lower()}", roc_auc[i])
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig("roc_curves.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("roc_curves.png")
        plt.close()
    
    def _log_prediction_confidence(self, y_pred_proba):
        """Log prediction confidence distribution."""
        max_proba = np.max(y_pred_proba, axis=1)
        
        plt.figure(figsize=(12, 5))
        
        # Histogram of confidence scores
        plt.subplot(1, 2, 1)
        plt.hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(max_proba.mean(), color='red', linestyle='--', 
                   label=f'Mean: {max_proba.mean():.3f}')
        plt.title('Distribution of Prediction Confidence')
        plt.xlabel('Max Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence by class
        plt.subplot(1, 2, 2)
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        for i, class_name in enumerate(class_names):
            plt.hist(y_pred_proba[:, i], bins=15, alpha=0.5, label=class_name)
        plt.title('Probability Distribution by Class')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("prediction_confidence.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("prediction_confidence.png")
        plt.close()
        
        # Log confidence metrics
        mlflow.log_metric("mean_prediction_confidence", max_proba.mean())
        mlflow.log_metric("min_prediction_confidence", max_proba.min())
        mlflow.log_metric("low_confidence_predictions", np.sum(max_proba < 0.8))
    
    def _log_data_distribution(self, X_train, y_train):
        """Log data distribution plots."""
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        
        # Create pairplot-style visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(feature_names):
            for class_idx in range(3):
                class_data = X_train[y_train == class_idx, i]
                axes[i].hist(class_data, alpha=0.6, label=class_names[class_idx], bins=15)
            
            axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(feature.replace("_", " ").title())
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("data_distribution.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("data_distribution.png")
        plt.close()
    
    def _log_detailed_classification_report(self, y_true, y_pred):
        """Log detailed classification report."""
        from sklearn.metrics import classification_report
        import json
        
        class_names = ['setosa', 'versicolor', 'virginica']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Log detailed metrics for each class
        for class_name in class_names:
            if class_name in report:
                class_metrics = report[class_name]
                for metric_name, value in class_metrics.items():
                    mlflow.log_metric(f"{class_name}_{metric_name}", value)
        
        # Log overall metrics
        for metric_name in ['accuracy', 'macro avg', 'weighted avg']:
            if metric_name in report:
                if metric_name == 'accuracy':
                    mlflow.log_metric("overall_accuracy", report[metric_name])
                else:
                    for sub_metric, value in report[metric_name].items():
                        mlflow.log_metric(f"{metric_name.replace(' ', '_')}_{sub_metric}", value)
        
        # Save detailed report as artifact
        with open("classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")
        os.remove("classification_report.json")
    
    def _log_model_interpretation(self):
        """Log model interpretation and explanations."""
        import json
        
        model_info = {
            "algorithm": "Random Forest",
            "algorithm_description": "Ensemble method using multiple decision trees",
            "hyperparameters": self.model.get_params(),
            "training_insights": {
                "feature_selection": "All 4 iris features used",
                "cross_validation": "5-fold cross-validation performed",
                "performance_category": "Classification task with 3 classes"
            },
            "model_characteristics": {
                "interpretability": "High - tree-based model",
                "overfitting_resistance": "Good - ensemble method",
                "training_speed": "Fast",
                "prediction_speed": "Fast"
            }
        }
        
        # Save model interpretation
        with open("model_interpretation.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact("model_interpretation.json")
        os.remove("model_interpretation.json")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")