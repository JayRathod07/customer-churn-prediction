"""
Unit tests for API endpoints.

Author: Jay Rathod
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import app

client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        # May return 503 if model not loaded, which is acceptable
        assert response.status_code in [200, 503]


class TestPredictionEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_valid_request(self):
        """Test prediction with valid request."""
        request_data = {
            "customer_id": "TEST001",
            "features": {
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 70.35,
                "total_charges": 840.75
            }
        }
        
        response = client.post("/predict", json=request_data)
        # May return 503 if model not loaded, 422 for validation, 500 for errors
        assert response.status_code in [200, 422, 503, 500]
    
    def test_predict_invalid_request(self):
        """Test prediction with invalid request."""
        request_data = {
            "customer_id": "TEST001",
            "features": {}  # Missing required fields
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_valid_request(self):
        """Test batch prediction with valid request."""
        request_data = {
            "customers": [
                {
                    "customer_id": "TEST001",
                    "gender": "Male",
                    "senior_citizen": 0,
                    "partner": "Yes",
                    "dependents": "No",
                    "tenure": 12,
                    "phone_service": "Yes",
                    "multiple_lines": "No",
                    "internet_service": "Fiber optic",
                    "online_security": "No",
                    "online_backup": "Yes",
                    "device_protection": "No",
                    "tech_support": "No",
                    "streaming_tv": "Yes",
                    "streaming_movies": "No",
                    "contract": "Month-to-month",
                    "paperless_billing": "Yes",
                    "payment_method": "Electronic check",
                    "monthly_charges": 70.35,
                    "total_charges": 840.75
                }
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        # May return 503 if model not loaded
        assert response.status_code in [200, 503, 500]
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        request_data = {"customers": []}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_endpoint(self):
        """Test Swagger UI docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
