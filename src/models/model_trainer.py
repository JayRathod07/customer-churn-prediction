"""
Model Training Module

This module provides functionality for training machine learning models
for customer churn prediction.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import os
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingError, ConfigurationError

logger = get_logger(__name__)


class ModelTrainer:
    """
    Handles training of multiple machine learning models with hyperparameter tuning.
    
    Features:
    - Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
    - Hyperparameter tuning with RandomizedSearchCV or GridSearchCV
    - Model selection based on specified metric
    - Model persistence with metadata
    - Train-test split with stratification
    
    Example:
        >>> trainer = ModelTrainer(config)
        >>> X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(X, y)
        >>> results = trainer.train_all_models(X_train, y_train)
        >>> best_model = trainer.select_best_model(results)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Configuration dictionary containing training parameters
            
        Raises:
            ConfigurationError: If required configuration is missing
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.storage_config = config.get('storage', {})
        
        # Validate configuration
        self._validate_config()
        
        # Initialize model registry
        self.models = {}
        self.trained_models = {}
        self.training_results = {}
        
        logger.info("ModelTrainer initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate that required configuration is present."""
        required_keys = ['test_size', 'random_state', 'models']
        for key in required_keys:
            if key not in self.training_config:
                raise ConfigurationError(f"Missing required training config: {key}")
    
    def prepare_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with optional stratification.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing (default from config)
            random_state: Random seed (default from config)
            stratify: Whether to stratify split based on target variable
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Raises:
            ModelTrainingError: If split fails
        """
        try:
            test_size = test_size or self.training_config.get('test_size', 0.2)
            random_state = random_state or self.training_config.get('random_state', 42)
            
            logger.info(f"Splitting data: test_size={test_size}, stratify={stratify}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if stratify else None
            )
            
            logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
            logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Failed to split data: {str(e)}")
            raise ModelTrainingError(f"Data split failed: {str(e)}")
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing trained model and metadata
        """
        logger.info("Training Logistic Regression model")
        
        model_config = self.training_config['models'].get('logistic_regression', {})
        
        if not model_config.get('enabled', True):
            logger.info("Logistic Regression is disabled in config")
            return None
        
        try:
            if tune_hyperparameters and 'hyperparameters' in model_config:
                # Hyperparameter tuning
                base_model = LogisticRegression(random_state=self.training_config['random_state'])
                param_grid = model_config['hyperparameters']
                
                model = self._tune_hyperparameters(
                    base_model, param_grid, X_train, y_train, 'logistic_regression'
                )
            else:
                # Train with default parameters
                model = LogisticRegression(random_state=self.training_config['random_state'])
                model.fit(X_train, y_train)
            
            result = {
                'model': model,
                'model_type': 'logistic_regression',
                'best_params': model.get_params() if not tune_hyperparameters else model.best_params_,
                'training_time': datetime.now().isoformat()
            }
            
            self.trained_models['logistic_regression'] = result
            logger.info("Logistic Regression training completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Logistic Regression training failed: {str(e)}")
            raise ModelTrainingError(f"Logistic Regression training failed: {str(e)}")
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing trained model and metadata
        """
        logger.info("Training Random Forest model")
        
        model_config = self.training_config['models'].get('random_forest', {})
        
        if not model_config.get('enabled', True):
            logger.info("Random Forest is disabled in config")
            return None
        
        try:
            if tune_hyperparameters and 'hyperparameters' in model_config:
                # Hyperparameter tuning
                base_model = RandomForestClassifier(random_state=self.training_config['random_state'])
                param_grid = model_config['hyperparameters']
                
                model = self._tune_hyperparameters(
                    base_model, param_grid, X_train, y_train, 'random_forest'
                )
            else:
                # Train with default parameters
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.training_config['random_state']
                )
                model.fit(X_train, y_train)
            
            result = {
                'model': model,
                'model_type': 'random_forest',
                'best_params': model.get_params() if not tune_hyperparameters else model.best_params_,
                'training_time': datetime.now().isoformat()
            }
            
            self.trained_models['random_forest'] = result
            logger.info("Random Forest training completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {str(e)}")
            raise ModelTrainingError(f"Random Forest training failed: {str(e)}")
    
    def train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train Gradient Boosting model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing trained model and metadata
        """
        logger.info("Training Gradient Boosting model")
        
        model_config = self.training_config['models'].get('gradient_boosting', {})
        
        if not model_config.get('enabled', True):
            logger.info("Gradient Boosting is disabled in config")
            return None
        
        try:
            if tune_hyperparameters and 'hyperparameters' in model_config:
                # Hyperparameter tuning
                base_model = GradientBoostingClassifier(random_state=self.training_config['random_state'])
                param_grid = model_config['hyperparameters']
                
                model = self._tune_hyperparameters(
                    base_model, param_grid, X_train, y_train, 'gradient_boosting'
                )
            else:
                # Train with default parameters
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.training_config['random_state']
                )
                model.fit(X_train, y_train)
            
            result = {
                'model': model,
                'model_type': 'gradient_boosting',
                'best_params': model.get_params() if not tune_hyperparameters else model.best_params_,
                'training_time': datetime.now().isoformat()
            }
            
            self.trained_models['gradient_boosting'] = result
            logger.info("Gradient Boosting training completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {str(e)}")
            raise ModelTrainingError(f"Gradient Boosting training failed: {str(e)}")
    
    def _tune_hyperparameters(
        self,
        base_model: Any,
        param_grid: Dict[str, List],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str
    ) -> Any:
        """
        Perform hyperparameter tuning using RandomizedSearchCV or GridSearchCV.
        
        Args:
            base_model: Base model to tune
            param_grid: Parameter grid for tuning
            X_train: Training features
            y_train: Training target
            model_name: Name of the model for logging
            
        Returns:
            Tuned model (best estimator)
        """
        tuning_config = self.training_config.get('hyperparameter_tuning', {})
        method = tuning_config.get('method', 'random_search')
        cv_folds = self.training_config.get('cv_folds', 5)
        scoring = self.training_config.get('scoring_metric', 'f1')
        
        logger.info(f"Tuning {model_name} using {method}")
        logger.info(f"Parameter grid: {param_grid}")
        
        try:
            if method == 'random_search':
                n_iter = tuning_config.get('n_iter', 20)
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=tuning_config.get('n_jobs', -1),
                    random_state=self.training_config['random_state'],
                    verbose=1
                )
            else:  # grid_search
                search = GridSearchCV(
                    base_model,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=tuning_config.get('n_jobs', -1),
                    verbose=1
                )
            
            search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {search.best_params_}")
            logger.info(f"Best {scoring} score: {search.best_score_:.4f}")
            
            return search
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed for {model_name}: {str(e)}")
            raise ModelTrainingError(f"Hyperparameter tuning failed: {str(e)}")
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all enabled models.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of trained models with metadata
        """
        logger.info("Training all enabled models")
        
        results = {}
        
        # Train Logistic Regression
        lr_result = self.train_logistic_regression(X_train, y_train, tune_hyperparameters)
        if lr_result:
            results['logistic_regression'] = lr_result
        
        # Train Random Forest
        rf_result = self.train_random_forest(X_train, y_train, tune_hyperparameters)
        if rf_result:
            results['random_forest'] = rf_result
        
        # Train Gradient Boosting
        gb_result = self.train_gradient_boosting(X_train, y_train, tune_hyperparameters)
        if gb_result:
            results['gradient_boosting'] = gb_result
        
        self.training_results = results
        logger.info(f"Trained {len(results)} models successfully")
        
        return results
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Get the actual model if it's a search object
            if hasattr(model, 'best_estimator_'):
                model = model.best_estimator_
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise ModelTrainingError(f"Model evaluation failed: {str(e)}")
    
    def select_best_model(
        self,
        results: Dict[str, Dict[str, Any]],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metric: str = 'f1'
    ) -> Tuple[str, Any, Dict[str, float]]:
        """
        Select the best model based on specified metric.
        
        Args:
            results: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            metric: Metric to use for selection (default: 'f1')
            
        Returns:
            Tuple of (model_name, model, metrics)
        """
        logger.info(f"Selecting best model based on {metric}")
        
        best_score = -1
        best_model_name = None
        best_model = None
        best_metrics = None
        
        for model_name, result in results.items():
            model = result['model']
            metrics = self.evaluate_model(model, X_test, y_test)
            
            logger.info(f"{model_name} - {metric}: {metrics[metric]:.4f}")
            
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
                best_model = model
                best_metrics = metrics
        
        logger.info(f"Best model: {best_model_name} with {metric}={best_score:.4f}")
        
        return best_model_name, best_model, best_metrics
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save trained model with metadata.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            metrics: Evaluation metrics
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        try:
            # Get the actual model if it's a search object
            if hasattr(model, 'best_estimator_'):
                actual_model = model.best_estimator_
                best_params = model.best_params_
            else:
                actual_model = model
                best_params = model.get_params()
            
            # Create models directory
            models_dir = self.storage_config.get('models_dir', 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Generate version timestamp
            version = datetime.now().strftime(self.storage_config.get('model_version_format', '%Y%m%d_%H%M%S'))
            
            # Save model
            model_filename = f"{model_name}_{version}.joblib"
            model_path = os.path.join(models_dir, model_filename)
            joblib.dump(actual_model, model_path)
            
            # Save metadata
            metadata_dict = {
                'model_name': model_name,
                'version': version,
                'training_time': datetime.now().isoformat(),
                'metrics': metrics,
                'best_params': best_params,
                'model_path': model_path
            }
            
            if metadata:
                metadata_dict.update(metadata)
            
            metadata_filename = f"{model_name}_{version}_metadata.json"
            metadata_path = os.path.join(models_dir, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Metadata saved: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise ModelTrainingError(f"Model save failed: {str(e)}")
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelTrainingError(f"Model load failed: {str(e)}")
