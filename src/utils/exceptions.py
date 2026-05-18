"""Custom exception classes for the churn prediction system."""


class ChurnPredictionError(Exception):
    """Base exception for all churn prediction errors."""
    pass


class DataLoadError(ChurnPredictionError):
    """Raised when data loading fails."""
    pass


class DataValidationError(ChurnPredictionError):
    """Raised when data validation fails."""
    pass


class FeatureEngineeringError(ChurnPredictionError):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(ChurnPredictionError):
    """Raised when model training fails."""
    pass


class ModelEvaluationError(ChurnPredictionError):
    """Raised when model evaluation fails."""
    pass


class ModelLoadError(ChurnPredictionError):
    """Raised when model loading fails."""
    pass


class PredictionError(ChurnPredictionError):
    """Raised when prediction fails."""
    pass


class ConfigurationError(ChurnPredictionError):
    """Raised when configuration is invalid."""
    pass
