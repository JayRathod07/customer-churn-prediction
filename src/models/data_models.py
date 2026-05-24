"""Pydantic data models for customer churn prediction.

This module provides data validation models for:
- Customer data input
- Training data
- API requests and responses
- Model metadata and evaluation metrics

Author: Jay Rathod
Date: 2024
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# Enums for categorical fields
class Gender(str, Enum):
    """Gender options."""
    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    """Yes/No options."""
    YES = "Yes"
    NO = "No"


class InternetService(str, Enum):
    """Internet service options."""
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class Contract(str, Enum):
    """Contract type options."""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethod(str, Enum):
    """Payment method options."""
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


# Customer Data Models
class CustomerData(BaseModel):
    """Model for raw customer data input.
    
    This model validates customer data before feature engineering.
    """
    
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: Gender = Field(..., description="Customer gender")
    senior_citizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0 or 1)")
    partner: YesNo = Field(..., description="Whether customer has a partner")
    dependents: YesNo = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=72, description="Number of months with company")
    phone_service: YesNo = Field(..., description="Whether customer has phone service")
    multiple_lines: str = Field(..., description="Whether customer has multiple lines")
    internet_service: InternetService = Field(..., description="Type of internet service")
    online_security: str = Field(..., description="Whether customer has online security")
    online_backup: str = Field(..., description="Whether customer has online backup")
    device_protection: str = Field(..., description="Whether customer has device protection")
    tech_support: str = Field(..., description="Whether customer has tech support")
    streaming_tv: str = Field(..., description="Whether customer has streaming TV")
    streaming_movies: str = Field(..., description="Whether customer has streaming movies")
    contract: Contract = Field(..., description="Contract type")
    paperless_billing: YesNo = Field(..., description="Whether customer has paperless billing")
    payment_method: PaymentMethod = Field(..., description="Payment method")
    monthly_charges: float = Field(..., gt=0, description="Monthly charges in dollars")
    total_charges: float = Field(..., ge=0, description="Total charges in dollars")
    churn: Optional[YesNo] = Field(None, description="Whether customer churned (for training data)")
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "customer_id": "CUST000001",
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 105.97,
                "total_charges": 1502.26,
                "churn": "Yes"
            }
        }
    )


class TrainingData(BaseModel):
    """Model for training dataset.
    
    This model represents a collection of customer records for training.
    """
    
    customers: List[CustomerData] = Field(..., description="List of customer records")
    total_records: int = Field(..., ge=1, description="Total number of records")
    churn_rate: float = Field(..., ge=0, le=1, description="Overall churn rate")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


# API Request/Response Models
class PredictionRequest(BaseModel):
    """Model for single prediction API request."""
    
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    gender: Gender
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(..., ge=0, le=72)
    phone_service: YesNo
    multiple_lines: str
    internet_service: InternetService
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: Contract
    paperless_billing: YesNo
    payment_method: PaymentMethod
    monthly_charges: float = Field(..., gt=0)
    total_charges: float = Field(..., ge=0)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "customer_id": "CUST000001",
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 105.97,
                "total_charges": 1502.26
            }
        }
    )


class PredictionResponse(BaseModel):
    """Model for single prediction API response."""
    
    customer_id: Optional[str] = Field(None, description="Customer identifier if provided")
    prediction: int = Field(..., ge=0, le=1, description="Churn prediction (0=No, 1=Yes)")
    probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: Literal["Low", "Medium", "High"] = Field("Low", description="Risk level based on probability")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST000001",
                "prediction": 1,
                "probability": 0.85,
                "risk_level": "High",
                "timestamp": "2024-05-20T10:30:00"
            }
        }
    )
    
    def __init__(self, **data):
        """Initialize and auto-determine risk level."""
        super().__init__(**data)
        # Determine risk level based on probability
        if self.probability < 0.3:
            object.__setattr__(self, 'risk_level', "Low")
        elif self.probability < 0.7:
            object.__setattr__(self, 'risk_level', "Medium")
        else:
            object.__setattr__(self, 'risk_level', "High")


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction API request."""
    
    customers: List[PredictionRequest] = Field(..., min_length=1, max_length=1000, 
                                               description="List of customers (max 1000)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customers": [
                    {
                        "customer_id": "CUST000001",
                        "gender": "Male",
                        "senior_citizen": 0,
                        "partner": "Yes",
                        "dependents": "No",
                        "tenure": 12,
                        "phone_service": "Yes",
                        "multiple_lines": "No",
                        "internet_service": "Fiber optic",
                        "online_security": "Yes",
                        "online_backup": "No",
                        "device_protection": "Yes",
                        "tech_support": "No",
                        "streaming_tv": "Yes",
                        "streaming_movies": "No",
                        "contract": "Month-to-month",
                        "paperless_billing": "Yes",
                        "payment_method": "Electronic check",
                        "monthly_charges": 105.97,
                        "total_charges": 1502.26
                    }
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction API response."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_predictions: int = Field(..., ge=0, description="Total number of predictions")
    high_risk_count: int = Field(..., ge=0, description="Number of high-risk customers")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")


# Model Metadata Models
class ModelMetadata(BaseModel):
    """Model for ML model metadata."""
    
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Model name (e.g., 'LogisticRegression', 'RandomForest')")
    model_version: str = Field(..., description="Model version (e.g., 'v1.0.0')")
    created_at: datetime = Field(default_factory=datetime.now, description="Model creation timestamp")
    trained_on: int = Field(..., ge=1, description="Number of training samples")
    features_count: int = Field(..., ge=1, description="Number of features")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    training_time_seconds: float = Field(..., ge=0, description="Training time in seconds")
    file_path: str = Field(..., description="Path to saved model file")
    transformer_path: Optional[str] = Field(None, description="Path to feature transformer file")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow fields starting with 'model_'
        json_schema_extra={
            "example": {
                "model_id": "model_20240520_103000",
                "model_name": "RandomForest",
                "model_version": "v1.0.0",
                "created_at": "2024-05-20T10:30:00",
                "trained_on": 800,
                "features_count": 35,
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                },
                "training_time_seconds": 5.23,
                "file_path": "models/random_forest_v1.0.0.joblib",
                "transformer_path": "artifacts/feature_transformer.joblib"
            }
        }
    )


class EvaluationMetrics(BaseModel):
    """Model for model evaluation metrics."""
    
    model_id: str = Field(..., description="Model identifier")
    accuracy: float = Field(..., ge=0, le=1, description="Accuracy score")
    precision: float = Field(..., ge=0, le=1, description="Precision score")
    recall: float = Field(..., ge=0, le=1, description="Recall score")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    roc_auc: float = Field(..., ge=0, le=1, description="ROC-AUC score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix [[TN, FP], [FN, TP]]")
    classification_report: Dict[str, Any] = Field(..., description="Detailed classification report")
    evaluated_at: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    test_samples: int = Field(..., ge=1, description="Number of test samples")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow fields starting with 'model_'
        json_schema_extra={
            "example": {
                "model_id": "model_20240520_103000",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.79,
                "f1_score": 0.80,
                "roc_auc": 0.88,
                "confusion_matrix": [[90, 15], [20, 75]],
                "classification_report": {
                    "0": {"precision": 0.82, "recall": 0.86, "f1-score": 0.84},
                    "1": {"precision": 0.83, "recall": 0.79, "f1-score": 0.81}
                },
                "evaluated_at": "2024-05-20T10:35:00",
                "test_samples": 200
            }
        }
    )


class HealthResponse(BaseModel):
    """Model for health check API response."""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow fields starting with 'model_'
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-05-20T10:30:00",
                "version": "1.0.0",
                "model_loaded": True,
                "model_version": "v1.0.0"
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Model for model info API response."""
    
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    evaluation_metrics: Optional[EvaluationMetrics] = Field(None, description="Evaluation metrics if available")
    feature_names: List[str] = Field(..., description="List of feature names")
    
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow fields starting with 'model_'
        json_schema_extra={
            "example": {
                "model_metadata": {
                    "model_id": "model_20240520_103000",
                    "model_name": "RandomForest",
                    "model_version": "v1.0.0",
                    "created_at": "2024-05-20T10:30:00",
                    "trained_on": 800,
                    "features_count": 35,
                    "hyperparameters": {"n_estimators": 100},
                    "training_time_seconds": 5.23,
                    "file_path": "models/random_forest_v1.0.0.joblib"
                },
                "evaluation_metrics": None,
                "feature_names": ["tenure", "monthly_charges", "total_charges"]
            }
        }
    )


class ErrorResponse(BaseModel):
    """Model for API error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "Field 'tenure' must be between 0 and 72",
                "timestamp": "2024-05-20T10:30:00"
            }
        }
    )
