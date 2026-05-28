"""
API package for customer churn prediction.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

from src.api.app import app, start_server
from src.api.prediction_service import PredictionService

__all__ = ['app', 'start_server', 'PredictionService']
