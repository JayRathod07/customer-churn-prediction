"""
FastAPI Application for Customer Churn Prediction

This module provides REST API endpoints for churn prediction.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
import os

from src.models.data_models import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse, ErrorResponse
)
from src.api.prediction_service import PredictionService
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load configuration
config_manager = ConfigManager()
config = config_manager.config
api_config = config.get('api', {})

# Configure CORS
if api_config.get('cors_enabled', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.get('cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize prediction service
prediction_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global prediction_service
    try:
        prediction_service = PredictionService(config)
        logger.info("Prediction service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    try:
        # Check if model is loaded
        model_loaded = prediction_service is not None and prediction_service.model is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            version="1.0.0"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        ModelInfoResponse with model details
    """
    try:
        if prediction_service is None or prediction_service.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = prediction_service.get_model_info()
        return ModelInfoResponse(**info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single churn prediction.
    
    Args:
        request: PredictionRequest with customer features
        
    Returns:
        PredictionResponse with prediction and probability
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        result = prediction_service.predict_single(request)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch churn predictions.
    
    Args:
        request: BatchPredictionRequest with list of customers
        
    Returns:
        BatchPredictionResponse with predictions for all customers
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )
        
        # Check batch size limit
        max_batch_size = api_config.get('max_batch_size', 1000)
        if len(request.customers) > max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum of {max_batch_size}"
            )
        
        result = prediction_service.predict_batch(request)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def start_server(host: str = None, port: int = None, reload: bool = False):
    """
    Start the API server.
    
    Args:
        host: Host address (default from config)
        port: Port number (default from config)
        reload: Enable auto-reload for development
    """
    host = host or api_config.get('host', '0.0.0.0')
    port = port or api_config.get('port', 8000)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=api_config.get('log_level', 'info')
    )


if __name__ == "__main__":
    start_server(reload=True)
