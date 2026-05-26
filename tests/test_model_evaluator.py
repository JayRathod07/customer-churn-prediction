"""
Unit tests for ModelEvaluator class.

Author: Jay Rathod
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.model_evaluator import ModelEvaluator
from src.utils.exceptions import ModelTrainingError


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'generate_plots': ['confusion_matrix', 'roc_curve', 'precision_recall_curve', 'feature_importance'],
            'report_format': 'markdown'
        },
        'storage': {
            'reports_dir': 'test_reports'
        }
    }


@pytest.fixture
def sample_data():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y = sample_data
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model


@pytest.fixture
def evaluator(sample_config):
    """Create ModelEvaluator instance."""
    return ModelEvaluator(sample_config)


@pytest.fixture
def temp_reports_dir():
    """Create temporary directory for reports."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestModelEvaluatorInitialization:
    """Test ModelEvaluator initialization."""
    
    def test_initialization_success(self, sample_config):
        """Test successful initialization."""
        evaluator = ModelEvaluator(sample_config)
        assert evaluator.config == sample_config
        assert evaluator.evaluation_config == sample_config['evaluation']
        assert evaluator.storage_config == sample_config['storage']
    
    def test_reports_directory_created(self, sample_config, temp_reports_dir):
        """Test that reports directory is created."""
        config = sample_config.copy()
        config['storage']['reports_dir'] = temp_reports_dir
        evaluator = ModelEvaluator(config)
        assert os.path.exists(temp_reports_dir)


class TestModelEvaluation:
    """Test model evaluation functionality."""
    
    def test_evaluate_model(self, evaluator, trained_model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        metrics = evaluator.evaluate(trained_model, X, y, model_name='test_model')
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Check metrics are in valid range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_evaluate_with_string_labels(self, evaluator, sample_data):
        """Test evaluation with string labels."""
        X, y = sample_data
        y_str = y.map({0: 'No', 1: 'Yes'})
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y_str)
        
        metrics = evaluator.evaluate(model, X, y_str, model_name='test_model')
        
        assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])


class TestConfusionMatrix:
    """Test confusion matrix functionality."""
    
    def test_compute_confusion_matrix(self, evaluator, trained_model, sample_data):
        """Test confusion matrix computation."""
        X, y = sample_data
        y_pred = trained_model.predict(X)
        
        cm = evaluator.compute_confusion_matrix(y, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y)
    
    def test_plot_confusion_matrix(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test confusion matrix plotting."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_pred = trained_model.predict(X)
        
        plot_path = evaluator.plot_confusion_matrix(y, y_pred)
        
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')
    
    def test_plot_confusion_matrix_custom_path(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test confusion matrix plotting with custom path."""
        X, y = sample_data
        y_pred = trained_model.predict(X)
        
        custom_path = os.path.join(temp_reports_dir, 'custom_cm.png')
        plot_path = evaluator.plot_confusion_matrix(y, y_pred, save_path=custom_path)
        
        assert plot_path == custom_path
        assert os.path.exists(custom_path)


class TestROCCurve:
    """Test ROC curve functionality."""
    
    def test_plot_roc_curve(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test ROC curve plotting."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        plot_path = evaluator.plot_roc_curve(y, y_pred_proba)
        
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')
    
    def test_plot_roc_curve_with_string_labels(self, evaluator, sample_data, temp_reports_dir):
        """Test ROC curve with string labels."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_str = y.map({0: 'No', 1: 'Yes'})
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y_str)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        plot_path = evaluator.plot_roc_curve(y_str, y_pred_proba)
        
        assert os.path.exists(plot_path)


class TestPrecisionRecallCurve:
    """Test Precision-Recall curve functionality."""
    
    def test_plot_pr_curve(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test PR curve plotting."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        plot_path = evaluator.plot_precision_recall_curve(y, y_pred_proba)
        
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')


class TestFeatureImportance:
    """Test feature importance functionality."""
    
    def test_extract_feature_importance_linear_model(self, evaluator, trained_model, sample_data):
        """Test feature importance extraction from linear model."""
        X, y = sample_data
        feature_names = list(X.columns)
        
        importance_df = evaluator.extract_feature_importance(trained_model, feature_names)
        
        assert not importance_df.empty
        assert len(importance_df) == len(feature_names)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_extract_feature_importance_tree_model(self, evaluator, sample_data):
        """Test feature importance extraction from tree model."""
        X, y = sample_data
        feature_names = list(X.columns)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importance_df = evaluator.extract_feature_importance(model, feature_names)
        
        assert not importance_df.empty
        assert len(importance_df) == len(feature_names)
    
    def test_plot_feature_importance(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test feature importance plotting."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        feature_names = list(X.columns)
        
        plot_path = evaluator.plot_feature_importance(trained_model, feature_names)
        
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')
    
    def test_plot_feature_importance_top_n(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test feature importance plotting with top N features."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        feature_names = list(X.columns)
        
        plot_path = evaluator.plot_feature_importance(trained_model, feature_names, top_n=5)
        
        assert os.path.exists(plot_path)


class TestClassificationReport:
    """Test classification report functionality."""
    
    def test_generate_classification_report(self, evaluator, trained_model, sample_data):
        """Test classification report generation."""
        X, y = sample_data
        y_pred = trained_model.predict(X)
        
        report = evaluator.generate_classification_report(y, y_pred)
        
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report


class TestReportGeneration:
    """Test report generation functionality."""
    
    def test_generate_markdown_report(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test Markdown report generation."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_pred = trained_model.predict(X)
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        metrics = evaluator.evaluate(trained_model, X, y)
        feature_names = list(X.columns)
        
        report_path = evaluator.generate_report(
            trained_model, 'test_model', metrics, X, y, 
            y_pred, y_pred_proba, feature_names, format='markdown'
        )
        
        assert os.path.exists(report_path)
        assert report_path.endswith('.md')
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Model Evaluation Report' in content
            assert 'Performance Metrics' in content
            assert 'Confusion Matrix' in content
    
    def test_generate_json_report(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test JSON report generation."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_pred = trained_model.predict(X)
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        metrics = evaluator.evaluate(trained_model, X, y)
        feature_names = list(X.columns)
        
        report_path = evaluator.generate_report(
            trained_model, 'test_model', metrics, X, y, 
            y_pred, y_pred_proba, feature_names, format='json'
        )
        
        assert os.path.exists(report_path)
        assert report_path.endswith('.json')
    
    def test_generate_report_without_feature_names(self, evaluator, trained_model, sample_data, temp_reports_dir):
        """Test report generation without feature names."""
        evaluator.reports_dir = temp_reports_dir
        
        X, y = sample_data
        y_pred = trained_model.predict(X)
        y_pred_proba = trained_model.predict_proba(X)[:, 1]
        
        metrics = evaluator.evaluate(trained_model, X, y)
        
        report_path = evaluator.generate_report(
            trained_model, 'test_model', metrics, X, y, 
            y_pred, y_pred_proba, feature_names=None, format='markdown'
        )
        
        assert os.path.exists(report_path)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_evaluate_empty_dataset(self, evaluator):
        """Test evaluation with empty dataset."""
        X = pd.DataFrame()
        y = pd.Series(dtype=int)
        model = LogisticRegression()
        
        with pytest.raises(Exception):
            evaluator.evaluate(model, X, y)
    
    def test_plot_with_invalid_data(self, evaluator):
        """Test plotting with invalid data."""
        y_true = pd.Series([])
        y_pred = np.array([])
        
        with pytest.raises(Exception):
            evaluator.plot_confusion_matrix(y_true, y_pred)
