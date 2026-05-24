"""Unit tests for Pydantic data models.

Comprehensive test suite covering data validation models for
customer data, API requests/responses, and model metadata.

Author: Jay Rathod
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models import (
    CustomerData,
    TrainingData,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelMetadata,
    EvaluationMetrics,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    Gender,
    YesNo,
    InternetService,
    Contract,
    PaymentMethod
)


class TestCustomerData:
    """Test suite for CustomerData model."""
    
    @pytest.fixture
    def valid_customer_data(self):
        """Create valid customer data."""
        return {
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
    
    def test_valid_customer_data(self, valid_customer_data):
        """Test creating CustomerData with valid data."""
        customer = CustomerData(**valid_customer_data)
        
        assert customer.customer_id == "CUST000001"
        assert customer.gender == Gender.MALE
        assert customer.senior_citizen == 0
        assert customer.tenure == 12
        assert customer.monthly_charges == 105.97
    
    def test_missing_required_field(self, valid_customer_data):
        """Test that missing required field raises error."""
        del valid_customer_data['customer_id']
        
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_data)
    
    def test_invalid_senior_citizen_value(self, valid_customer_data):
        """Test that invalid senior_citizen value raises error."""
        valid_customer_data['senior_citizen'] = 2
        
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_data)
    
    def test_invalid_tenure_value(self, valid_customer_data):
        """Test that invalid tenure value raises error."""
        valid_customer_data['tenure'] = 100
        
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_data)
    
    def test_negative_monthly_charges(self, valid_customer_data):
        """Test that negative monthly_charges raises error."""
        valid_customer_data['monthly_charges'] = -10.0
        
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_data)
    
    def test_optional_churn_field(self, valid_customer_data):
        """Test that churn field is optional."""
        del valid_customer_data['churn']
        
        customer = CustomerData(**valid_customer_data)
        assert customer.churn is None


class TestPredictionRequest:
    """Test suite for PredictionRequest model."""
    
    @pytest.fixture
    def valid_prediction_request(self):
        """Create valid prediction request."""
        return {
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
    
    def test_valid_prediction_request(self, valid_prediction_request):
        """Test creating PredictionRequest with valid data."""
        request = PredictionRequest(**valid_prediction_request)
        
        assert request.customer_id == "CUST000001"
        assert request.gender == Gender.MALE
        assert request.monthly_charges == 105.97
    
    def test_optional_customer_id(self, valid_prediction_request):
        """Test that customer_id is optional."""
        del valid_prediction_request['customer_id']
        
        request = PredictionRequest(**valid_prediction_request)
        assert request.customer_id is None


class TestPredictionResponse:
    """Test suite for PredictionResponse model."""
    
    def test_valid_prediction_response(self):
        """Test creating PredictionResponse with valid data."""
        response = PredictionResponse(
            customer_id="CUST000001",
            prediction=1,
            probability=0.85,
            risk_level="High"
        )
        
        assert response.customer_id == "CUST000001"
        assert response.prediction == 1
        assert response.probability == 0.85
        assert response.risk_level == "High"
        assert isinstance(response.timestamp, datetime)
    
    def test_risk_level_determination_low(self):
        """Test risk level determination for low probability."""
        response = PredictionResponse(
            prediction=0,
            probability=0.2,
            risk_level="Low"
        )
        
        assert response.risk_level == "Low"
    
    def test_risk_level_determination_medium(self):
        """Test risk level determination for medium probability."""
        response = PredictionResponse(
            prediction=1,
            probability=0.5,
            risk_level="Medium"
        )
        
        assert response.risk_level == "Medium"
    
    def test_risk_level_determination_high(self):
        """Test risk level determination for high probability."""
        response = PredictionResponse(
            prediction=1,
            probability=0.85,
            risk_level="High"
        )
        
        assert response.risk_level == "High"
    
    def test_invalid_prediction_value(self):
        """Test that invalid prediction value raises error."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction=2,
                probability=0.85,
                risk_level="High"
            )
    
    def test_invalid_probability_value(self):
        """Test that invalid probability value raises error."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction=1,
                probability=1.5,
                risk_level="High"
            )


class TestBatchPredictionRequest:
    """Test suite for BatchPredictionRequest model."""
    
    @pytest.fixture
    def valid_batch_request(self):
        """Create valid batch prediction request."""
        return {
            "customers": [
                {
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
    
    def test_valid_batch_request(self, valid_batch_request):
        """Test creating BatchPredictionRequest with valid data."""
        request = BatchPredictionRequest(**valid_batch_request)
        
        assert len(request.customers) == 1
        assert isinstance(request.customers[0], PredictionRequest)
    
    def test_empty_customers_list(self):
        """Test that empty customers list raises error."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(customers=[])


class TestBatchPredictionResponse:
    """Test suite for BatchPredictionResponse model."""
    
    def test_valid_batch_response(self):
        """Test creating BatchPredictionResponse with valid data."""
        predictions = [
            PredictionResponse(prediction=1, probability=0.85, risk_level="High"),
            PredictionResponse(prediction=0, probability=0.2, risk_level="Low")
        ]
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_predictions=2,
            high_risk_count=1,
            processing_time_ms=150.5
        )
        
        assert len(response.predictions) == 2
        assert response.total_predictions == 2
        assert response.high_risk_count == 1
        assert response.processing_time_ms == 150.5


class TestModelMetadata:
    """Test suite for ModelMetadata model."""
    
    def test_valid_model_metadata(self):
        """Test creating ModelMetadata with valid data."""
        metadata = ModelMetadata(
            model_id="model_20240520_103000",
            model_name="RandomForest",
            model_version="v1.0.0",
            trained_on=800,
            features_count=35,
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            training_time_seconds=5.23,
            file_path="models/random_forest_v1.0.0.joblib",
            transformer_path="artifacts/feature_transformer.joblib"
        )
        
        assert metadata.model_id == "model_20240520_103000"
        assert metadata.model_name == "RandomForest"
        assert metadata.trained_on == 800
        assert metadata.features_count == 35
        assert isinstance(metadata.created_at, datetime)
    
    def test_optional_transformer_path(self):
        """Test that transformer_path is optional."""
        metadata = ModelMetadata(
            model_id="model_20240520_103000",
            model_name="LogisticRegression",
            model_version="v1.0.0",
            trained_on=800,
            features_count=35,
            hyperparameters={},
            training_time_seconds=2.5,
            file_path="models/logistic_regression_v1.0.0.joblib"
        )
        
        assert metadata.transformer_path is None


class TestEvaluationMetrics:
    """Test suite for EvaluationMetrics model."""
    
    def test_valid_evaluation_metrics(self):
        """Test creating EvaluationMetrics with valid data."""
        metrics = EvaluationMetrics(
            model_id="model_20240520_103000",
            accuracy=0.85,
            precision=0.82,
            recall=0.79,
            f1_score=0.80,
            roc_auc=0.88,
            confusion_matrix=[[90, 15], [20, 75]],
            classification_report={
                "0": {"precision": 0.82, "recall": 0.86, "f1-score": 0.84},
                "1": {"precision": 0.83, "recall": 0.79, "f1-score": 0.81}
            },
            test_samples=200
        )
        
        assert metrics.model_id == "model_20240520_103000"
        assert metrics.accuracy == 0.85
        assert metrics.f1_score == 0.80
        assert len(metrics.confusion_matrix) == 2
        assert isinstance(metrics.evaluated_at, datetime)
    
    def test_invalid_metric_range(self):
        """Test that metric values outside [0, 1] raise error."""
        with pytest.raises(ValidationError):
            EvaluationMetrics(
                model_id="model_20240520_103000",
                accuracy=1.5,  # Invalid
                precision=0.82,
                recall=0.79,
                f1_score=0.80,
                roc_auc=0.88,
                confusion_matrix=[[90, 15], [20, 75]],
                classification_report={},
                test_samples=200
            )


class TestHealthResponse:
    """Test suite for HealthResponse model."""
    
    def test_valid_health_response(self):
        """Test creating HealthResponse with valid data."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            model_loaded=True,
            model_version="v1.0.0"
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.model_loaded is True
        assert response.model_version == "v1.0.0"
        assert isinstance(response.timestamp, datetime)
    
    def test_unhealthy_status(self):
        """Test creating HealthResponse with unhealthy status."""
        response = HealthResponse(
            status="unhealthy",
            version="1.0.0",
            model_loaded=False
        )
        
        assert response.status == "unhealthy"
        assert response.model_loaded is False


class TestErrorResponse:
    """Test suite for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test creating ErrorResponse with valid data."""
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid input data",
            detail="Field 'tenure' must be between 0 and 72"
        )
        
        assert response.error == "ValidationError"
        assert response.message == "Invalid input data"
        assert response.detail == "Field 'tenure' must be between 0 and 72"
        assert isinstance(response.timestamp, datetime)
    
    def test_optional_detail(self):
        """Test that detail field is optional."""
        response = ErrorResponse(
            error="InternalError",
            message="An unexpected error occurred"
        )
        
        assert response.detail is None


class TestTrainingData:
    """Test suite for TrainingData model."""
    
    def test_valid_training_data(self):
        """Test creating TrainingData with valid data."""
        customers = [
            CustomerData(
                customer_id="CUST000001",
                gender="Male",
                senior_citizen=0,
                partner="Yes",
                dependents="No",
                tenure=12,
                phone_service="Yes",
                multiple_lines="No",
                internet_service="Fiber optic",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="No",
                contract="Month-to-month",
                paperless_billing="Yes",
                payment_method="Electronic check",
                monthly_charges=105.97,
                total_charges=1502.26,
                churn="Yes"
            )
        ]
        
        training_data = TrainingData(
            customers=customers,
            total_records=1,
            churn_rate=1.0
        )
        
        assert len(training_data.customers) == 1
        assert training_data.total_records == 1
        assert training_data.churn_rate == 1.0
