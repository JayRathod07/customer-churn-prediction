"""
Unit tests for ModelTrainer class.

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.models.model_trainer import ModelTrainer
from src.utils.exceptions import ModelTrainingError, ConfigurationError


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'cv_folds': 3,
            'scoring_metric': 'f1',
            'models': {
                'logistic_regression': {
                    'enabled': True,
                    'hyperparameters': {
                        'C': [0.1, 1],
                        'penalty': ['l2'],
                        'solver': ['liblinear'],
                        'max_iter': [1000]
                    }
                },
                'random_forest': {
                    'enabled': True,
                    'hyperparameters': {
                        'n_estimators': [50, 100],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5],
                        'class_weight': ['balanced']
                    }
                },
                'gradient_boosting': {
                    'enabled': True,
                    'hyperparameters': {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5]
                    }
                }
            },
            'hyperparameter_tuning': {
                'method': 'random_search',
                'n_iter': 2,
                'n_jobs': 1
            }
        },
        'storage': {
            'models_dir': 'test_models',
            'model_version_format': '%Y%m%d_%H%M%S',
            'keep_n_models': 5
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
def trainer(sample_config):
    """Create ModelTrainer instance."""
    return ModelTrainer(sample_config)


@pytest.fixture
def temp_models_dir():
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestModelTrainerInitialization:
    """Test ModelTrainer initialization."""
    
    def test_initialization_success(self, sample_config):
        """Test successful initialization."""
        trainer = ModelTrainer(sample_config)
        assert trainer.config == sample_config
        assert trainer.training_config == sample_config['training']
        assert trainer.storage_config == sample_config['storage']
        assert isinstance(trainer.models, dict)
        assert isinstance(trainer.trained_models, dict)
    
    def test_initialization_missing_config(self):
        """Test initialization with missing required config."""
        invalid_config = {'training': {}}
        with pytest.raises(ConfigurationError):
            ModelTrainer(invalid_config)
    
    def test_initialization_partial_config(self, sample_config):
        """Test initialization with partial config."""
        # Remove optional keys
        config = sample_config.copy()
        config['training'].pop('cv_folds', None)
        trainer = ModelTrainer(config)
        assert trainer is not None


class TestTrainTestSplit:
    """Test train-test split functionality."""
    
    def test_basic_split(self, trainer, sample_data):
        """Test basic train-test split."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_split_with_stratification(self, trainer, sample_data):
        """Test split with stratification."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
            X, y, stratify=True
        )
        
        # Check class distribution is similar
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.1
    
    def test_split_custom_test_size(self, trainer, sample_data):
        """Test split with custom test size."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
            X, y, test_size=0.3
        )
        
        expected_test_size = int(len(X) * 0.3)
        assert abs(len(X_test) - expected_test_size) <= 1
    
    def test_split_reproducibility(self, trainer, sample_data):
        """Test that split is reproducible with same random state."""
        X, y = sample_data
        
        X_train1, X_test1, y_train1, y_test1 = trainer.prepare_train_test_split(
            X, y, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = trainer.prepare_train_test_split(
            X, y, random_state=42
        )
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)


class TestLogisticRegressionTraining:
    """Test Logistic Regression training."""
    
    def test_train_without_tuning(self, trainer, sample_data):
        """Test training without hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=False
        )
        
        assert result is not None
        assert 'model' in result
        assert 'model_type' in result
        assert result['model_type'] == 'logistic_regression'
        assert isinstance(result['model'], LogisticRegression)
    
    def test_train_with_tuning(self, trainer, sample_data):
        """Test training with hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=True
        )
        
        assert result is not None
        assert 'model' in result
        assert 'best_params' in result
        assert hasattr(result['model'], 'best_estimator_')
    
    def test_train_disabled_model(self, sample_config, sample_data):
        """Test training when model is disabled."""
        config = sample_config.copy()
        config['training']['models']['logistic_regression']['enabled'] = False
        trainer = ModelTrainer(config)
        
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(X_train, y_train)
        assert result is None


class TestRandomForestTraining:
    """Test Random Forest training."""
    
    def test_train_without_tuning(self, trainer, sample_data):
        """Test training without hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_random_forest(
            X_train, y_train, tune_hyperparameters=False
        )
        
        assert result is not None
        assert 'model' in result
        assert result['model_type'] == 'random_forest'
        assert isinstance(result['model'], RandomForestClassifier)
    
    def test_train_with_tuning(self, trainer, sample_data):
        """Test training with hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_random_forest(
            X_train, y_train, tune_hyperparameters=True
        )
        
        assert result is not None
        assert 'best_params' in result
        assert hasattr(result['model'], 'best_estimator_')


class TestGradientBoostingTraining:
    """Test Gradient Boosting training."""
    
    def test_train_without_tuning(self, trainer, sample_data):
        """Test training without hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_gradient_boosting(
            X_train, y_train, tune_hyperparameters=False
        )
        
        assert result is not None
        assert 'model' in result
        assert result['model_type'] == 'gradient_boosting'
        assert isinstance(result['model'], GradientBoostingClassifier)
    
    def test_train_with_tuning(self, trainer, sample_data):
        """Test training with hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_gradient_boosting(
            X_train, y_train, tune_hyperparameters=True
        )
        
        assert result is not None
        assert 'best_params' in result
        assert hasattr(result['model'], 'best_estimator_')


class TestTrainAllModels:
    """Test training all models."""
    
    def test_train_all_enabled_models(self, trainer, sample_data):
        """Test training all enabled models."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        results = trainer.train_all_models(
            X_train, y_train, tune_hyperparameters=False
        )
        
        assert len(results) == 3
        assert 'logistic_regression' in results
        assert 'random_forest' in results
        assert 'gradient_boosting' in results
    
    def test_train_all_with_tuning(self, trainer, sample_data):
        """Test training all models with hyperparameter tuning."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        results = trainer.train_all_models(
            X_train, y_train, tune_hyperparameters=True
        )
        
        assert len(results) > 0
        for model_name, result in results.items():
            assert 'model' in result
            assert 'best_params' in result


class TestModelEvaluation:
    """Test model evaluation."""
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        # Train a simple model
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=False
        )
        
        metrics = trainer.evaluate_model(result['model'], X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Check metrics are in valid range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_evaluate_tuned_model(self, trainer, sample_data):
        """Test evaluation of tuned model."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=True
        )
        
        metrics = trainer.evaluate_model(result['model'], X_test, y_test)
        
        assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])


class TestModelSelection:
    """Test model selection."""
    
    def test_select_best_model(self, trainer, sample_data):
        """Test selecting best model."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        results = trainer.train_all_models(
            X_train, y_train, tune_hyperparameters=False
        )
        
        best_name, best_model, best_metrics = trainer.select_best_model(
            results, X_test, y_test, metric='f1'
        )
        
        assert best_name in ['logistic_regression', 'random_forest', 'gradient_boosting']
        assert best_model is not None
        assert isinstance(best_metrics, dict)
        assert 'f1' in best_metrics
    
    def test_select_best_by_different_metrics(self, trainer, sample_data):
        """Test selecting best model by different metrics."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        results = trainer.train_all_models(
            X_train, y_train, tune_hyperparameters=False
        )
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            best_name, best_model, best_metrics = trainer.select_best_model(
                results, X_test, y_test, metric=metric
            )
            assert best_name is not None
            assert metric in best_metrics


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_save_model(self, trainer, sample_data, temp_models_dir):
        """Test saving a trained model."""
        # Update config to use temp directory
        trainer.storage_config['models_dir'] = temp_models_dir
        
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=False
        )
        
        metrics = trainer.evaluate_model(result['model'], X_test, y_test)
        
        model_path = trainer.save_model(
            result['model'],
            'logistic_regression',
            metrics
        )
        
        assert os.path.exists(model_path)
        assert model_path.endswith('.joblib')
        
        # Check metadata file exists
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        assert os.path.exists(metadata_path)
    
    def test_load_model(self, trainer, sample_data, temp_models_dir):
        """Test loading a saved model."""
        trainer.storage_config['models_dir'] = temp_models_dir
        
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=False
        )
        
        metrics = trainer.evaluate_model(result['model'], X_test, y_test)
        model_path = trainer.save_model(
            result['model'],
            'logistic_regression',
            metrics
        )
        
        # Load the model
        loaded_model = ModelTrainer.load_model(model_path)
        
        assert loaded_model is not None
        assert isinstance(loaded_model, LogisticRegression)
        
        # Test that loaded model can make predictions
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)
    
    def test_save_model_with_metadata(self, trainer, sample_data, temp_models_dir):
        """Test saving model with additional metadata."""
        trainer.storage_config['models_dir'] = temp_models_dir
        
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        
        result = trainer.train_logistic_regression(
            X_train, y_train, tune_hyperparameters=False
        )
        
        metrics = trainer.evaluate_model(result['model'], X_test, y_test)
        
        custom_metadata = {
            'author': 'Jay Rathod',
            'dataset_size': len(X_train),
            'features': list(X_train.columns)
        }
        
        model_path = trainer.save_model(
            result['model'],
            'logistic_regression',
            metrics,
            metadata=custom_metadata
        )
        
        assert os.path.exists(model_path)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self, trainer):
        """Test handling of empty dataset."""
        X = pd.DataFrame()
        y = pd.Series(dtype=int)
        
        with pytest.raises(Exception):
            trainer.prepare_train_test_split(X, y)
    
    def test_single_class_dataset(self, trainer):
        """Test handling of single-class dataset."""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series([0] * 100)
        
        # Should work but stratification might fail
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
            X, y, stratify=False
        )
        assert len(X_train) > 0
