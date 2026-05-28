"""
Prediction Service for Customer Churn Prediction

This module handles model loading and prediction logic.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import os
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from src.models.data_models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from src.features.feature_transformer import FeatureTransformer
from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingError

logger = get_logger(__name__)


class PredictionService:
    """
    Service for making churn predictions.
    
    Features:
    - Lazy loading of model and transformer
    - Single and batch predictions
    - Model metadata management
    - Error handling and validation
    
    Example:
        >>> service = PredictionService(config)
        >>> result = service.predict_single(request)
        >>> batch_result = service.predict_batch(batch_request)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PredictionService.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.storage_config = config.get('storage', {})
        
        self.model = None
        self.transformer = None
        self.model_metadata = None
        
        # Load model and transformer
        self._load_model()
        self._load_transformer()
        
        logger.info("PredictionService initialized successfully")
    
    def _load_model(self):
        """Load the latest trained model."""
        try:
            models_dir = self.storage_config.get('models_dir', 'models')
            
            if not os.path.exists(models_dir):
                logger.warning(f"Models directory not found: {models_dir}")
                return
            
            # Find latest model
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            
            if not model_files:
                logger.warning("No model files found")
                return
            
            # Sort by modification time (latest first)
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            latest_model = model_files[0]
            model_path = os.path.join(models_dir, latest_model)
            
            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            
            # Load metadata
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Model metadata loaded from: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelTrainingError(f"Model loading failed: {str(e)}")
    
    def _load_transformer(self):
        """Load the feature transformer."""
        try:
            artifacts_dir = self.storage_config.get('artifacts_dir', 'artifacts')
            transformer_path = os.path.join(artifacts_dir, 'feature_transformer.joblib')
            
            if not os.path.exists(transformer_path):
                logger.warning(f"Transformer not found: {transformer_path}")
                return
            
            self.transformer = FeatureTransformer.load(transformer_path)
            logger.info(f"Transformer loaded from: {transformer_path}")
            
        except Exception as e:
            logger.error(f"Failed to load transformer: {str(e)}")
            raise ModelTrainingError(f"Transformer loading failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                "model_name": "None",
                "model_version": "None",
                "model_type": "None",
                "training_date": "None",
                "metrics": {}
            }
        
        model_type = type(self.model).__name__
        
        info = {
            "model_name": self.model_metadata.get('model_name', 'unknown') if self.model_metadata else 'unknown',
            "model_version": self.model_metadata.get('version', 'unknown') if self.model_metadata else 'unknown',
            "model_type": model_type,
            "training_date": self.model_metadata.get('training_time', 'unknown') if self.model_metadata else 'unknown',
            "metrics": self.model_metadata.get('metrics', {}) if self.model_metadata else {}
        }
        
        return info
    
    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make a single prediction.
        
        Args:
            request: PredictionRequest with customer features
            
        Returns:
            PredictionResponse with prediction and probability
        """
        try:
            if self.model is None:
                raise ModelTrainingError("Model not loaded")
            
            if self.transformer is None:
                raise ModelTrainingError("Transformer not loaded")
            
            # Convert request to DataFrame
            features_dict = request.features
            df = pd.DataFrame([features_dict])
            
            # Transform features
            X_transformed = self.transformer.transform(df)
            
            # Make prediction
            prediction = self.model.predict(X_transformed)[0]
            probability = self.model.predict_proba(X_transformed)[0, 1]
            
            # Convert prediction to standard format
            if isinstance(prediction, (np.integer, int)):
                prediction_value = int(prediction)
            else:
                prediction_value = 1 if prediction == 'Yes' else 0
            
            # Create response
            response = PredictionResponse(
                customer_id=request.customer_id,
                prediction=prediction_value,
                probability=float(probability),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Prediction made for customer {request.customer_id}: {prediction_value} ({probability:.4f})")
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelTrainingError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Make batch predictions.
        
        Args:
            request: BatchPredictionRequest with list of customers
            
        Returns:
            BatchPredictionResponse with predictions for all customers
        """
        try:
            if self.model is None:
                raise ModelTrainingError("Model not loaded")
            
            if self.transformer is None:
                raise ModelTrainingError("Transformer not loaded")
            
            predictions = []
            
            for customer_request in request.customers:
                try:
                    pred_request = PredictionRequest(
                        customer_id=customer_request.customer_id,
                        features=customer_request.model_dump(exclude={'customer_id', 'churn'})
                    )
                    result = self.predict_single(pred_request)
                    predictions.append(result)
                except Exception as e:
                    logger.error(f"Failed to predict for customer {customer_request.customer_id}: {str(e)}")
                    # Add error prediction
                    predictions.append(
                        PredictionResponse(
                            customer_id=customer_request.customer_id,
                            prediction=0,
                            probability=0.0,
                            timestamp=datetime.now().isoformat()
                        )
                    )
            
            response = BatchPredictionResponse(
                predictions=predictions,
                total_count=len(predictions),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Batch prediction completed for {len(predictions)} customers")
            
            return response
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise ModelTrainingError(f"Batch prediction failed: {str(e)}")
