"""
Model Evaluation Module

This module provides functionality for evaluating machine learning models
with comprehensive metrics, visualizations, and reports.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    auc
)

from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingError

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Handles comprehensive evaluation of machine learning models.
    
    Features:
    - Compute classification metrics
    - Generate confusion matrix
    - Extract feature importance
    - Create visualizations (ROC curve, PR curve, confusion matrix)
    - Generate evaluation reports (Markdown, JSON, HTML)
    
    Example:
        >>> evaluator = ModelEvaluator(config)
        >>> metrics = evaluator.evaluate(model, X_test, y_test)
        >>> evaluator.plot_confusion_matrix(y_test, y_pred)
        >>> evaluator.plot_roc_curve(y_test, y_pred_proba)
        >>> report = evaluator.generate_report(model, metrics, feature_names)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator with configuration.
        
        Args:
            config: Configuration dictionary containing evaluation parameters
        """
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.storage_config = config.get('storage', {})
        
        # Set up directories
        self.reports_dir = self.storage_config.get('reports_dir', 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set plot style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        
        logger.info("ModelEvaluator initialized successfully")
    
    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a trained model with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_name}")
            
            # Get the actual model if it's a search object
            if hasattr(model, 'best_estimator_'):
                model = model.best_estimator_
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Handle string labels
            pos_label = 1
            if y_test.dtype == 'object' or isinstance(y_test.iloc[0], str):
                pos_label = 'Yes'
            
            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, pos_label=pos_label),
                'recall': recall_score(y_test, y_pred, pos_label=pos_label),
                'f1': f1_score(y_test, y_pred, pos_label=pos_label),
                'roc_auc': roc_auc_score(
                    (y_test == pos_label).astype(int), 
                    y_pred_proba
                )
            }
            
            logger.info(f"{model_name} evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise ModelTrainingError(f"Model evaluation failed: {str(e)}")
    
    def compute_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion matrix computed: shape {cm.shape}")
        return cm
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Confusion Matrix"
    ) -> str:
        """
        Plot confusion matrix with annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        try:
            cm = self.compute_confusion_matrix(y_true, y_pred)
            
            # Get unique labels
            labels = sorted(y_true.unique())
            
            # Create figure
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'}
            )
            plt.title(title, fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.reports_dir, f'confusion_matrix_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {str(e)}")
            raise ModelTrainingError(f"Confusion matrix plot failed: {str(e)}")
    
    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "ROC Curve"
    ) -> str:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        try:
            # Handle string labels
            pos_label = 1
            if y_true.dtype == 'object' or isinstance(y_true.iloc[0], str):
                pos_label = 'Yes'
                y_true_binary = (y_true == pos_label).astype(int)
            else:
                y_true_binary = y_true
            
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.reports_dir, f'roc_curve_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {str(e)}")
            raise ModelTrainingError(f"ROC curve plot failed: {str(e)}")
    
    def plot_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Precision-Recall Curve"
    ) -> str:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot (optional)
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        try:
            # Handle string labels
            pos_label = 1
            if y_true.dtype == 'object' or isinstance(y_true.iloc[0], str):
                pos_label = 'Yes'
                y_true_binary = (y_true == pos_label).astype(int)
            else:
                y_true_binary = y_true
            
            # Compute PR curve
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(loc="lower left", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.reports_dir, f'pr_curve_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision-Recall curve plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot PR curve: {str(e)}")
            raise ModelTrainingError(f"PR curve plot failed: {str(e)}")
    
    def extract_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature names and importance scores
        """
        try:
            # Get the actual model if it's a search object
            if hasattr(model, 'best_estimator_'):
                model = model.best_estimator_
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
            else:
                logger.warning(f"Model {type(model).__name__} does not support feature importance")
                return pd.DataFrame()
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            logger.info(f"Extracted feature importance for {len(feature_names)} features")
            return importance_df
            
        except Exception as e:
            logger.error(f"Failed to extract feature importance: {str(e)}")
            return pd.DataFrame()
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        save_path: Optional[str] = None,
        title: str = "Feature Importance"
    ) -> str:
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save the plot (optional)
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        try:
            importance_df = self.extract_feature_importance(model, feature_names)
            
            if importance_df.empty:
                logger.warning("No feature importance to plot")
                return ""
            
            # Get top N features
            top_features = importance_df.head(top_n)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.reports_dir, f'feature_importance_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {str(e)}")
            raise ModelTrainingError(f"Feature importance plot failed: {str(e)}")
    
    def generate_classification_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred)
        logger.info("Classification report generated")
        return report
    
    def generate_report(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        feature_names: Optional[List[str]] = None,
        format: str = 'markdown'
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Evaluation metrics
            X_test: Test features
            y_test: Test target
            y_pred: Predictions
            y_pred_proba: Prediction probabilities
            feature_names: List of feature names (optional)
            format: Report format ('markdown', 'json', 'html')
            
        Returns:
            Path to saved report
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'markdown':
                return self._generate_markdown_report(
                    model, model_name, metrics, X_test, y_test, 
                    y_pred, y_pred_proba, feature_names, timestamp
                )
            elif format == 'json':
                return self._generate_json_report(
                    model_name, metrics, X_test, y_test, 
                    y_pred, feature_names, timestamp
                )
            else:
                logger.warning(f"Unsupported format: {format}. Using markdown.")
                return self._generate_markdown_report(
                    model, model_name, metrics, X_test, y_test, 
                    y_pred, y_pred_proba, feature_names, timestamp
                )
                
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise ModelTrainingError(f"Report generation failed: {str(e)}")
    
    def _generate_markdown_report(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        feature_names: Optional[List[str]],
        timestamp: str
    ) -> str:
        """Generate Markdown format report."""
        
        # Generate plots
        cm_path = self.plot_confusion_matrix(y_test, y_pred)
        roc_path = self.plot_roc_curve(y_test, y_pred_proba)
        pr_path = self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        fi_path = ""
        if feature_names:
            fi_path = self.plot_feature_importance(model, feature_names)
        
        # Generate classification report
        class_report = self.generate_classification_report(y_test, y_pred)
        
        # Create report content
        report_content = f"""# Model Evaluation Report

**Model**: {model_name}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Test Set Size**: {len(X_test)} samples

---

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-Score | {metrics['f1']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |

---

## Classification Report

```
{class_report}
```

---

## Visualizations

### Confusion Matrix
![Confusion Matrix]({os.path.basename(cm_path)})

### ROC Curve
![ROC Curve]({os.path.basename(roc_path)})

### Precision-Recall Curve
![PR Curve]({os.path.basename(pr_path)})

"""
        
        if fi_path:
            report_content += f"""### Feature Importance
![Feature Importance]({os.path.basename(fi_path)})

"""
        
        report_content += f"""---

## Model Details

- **Model Type**: {type(model).__name__ if not hasattr(model, 'best_estimator_') else type(model.best_estimator_).__name__}
- **Number of Features**: {X_test.shape[1]}
- **Test Set Size**: {len(X_test)}
- **Class Distribution**: {dict(y_test.value_counts())}

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: Jay Rathod
"""
        
        # Save report
        report_path = os.path.join(self.reports_dir, f'evaluation_report_{model_name}_{timestamp}.md')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report saved to: {report_path}")
        return report_path
    
    def _generate_json_report(
        self,
        model_name: str,
        metrics: Dict[str, float],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]],
        timestamp: str
    ) -> str:
        """Generate JSON format report."""
        
        # Convert numpy/pandas types to Python native types for JSON serialization
        class_dist = y_test.value_counts().to_dict()
        class_dist_serializable = {str(k): int(v) for k, v in class_dist.items()}
        
        report_data = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'test_set_size': int(len(X_test)),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'class_distribution': class_dist_serializable,
            'confusion_matrix': self.compute_confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Save report
        report_path = os.path.join(self.reports_dir, f'evaluation_report_{model_name}_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report saved to: {report_path}")
        return report_path
