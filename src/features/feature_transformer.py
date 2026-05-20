"""Feature engineering module for customer churn prediction.

This module provides the FeatureTransformer class for encoding categorical variables,
scaling numerical features, and creating derived features.

Author: Jay Rathod
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """Transforms raw customer data into ML-ready features.
    
    This class handles:
    - Categorical encoding (one-hot and binary)
    - Numerical scaling (StandardScaler)
    - Derived feature creation
    - Transformer persistence
    """
    
    def __init__(self):
        """Initialize the FeatureTransformer."""
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        # Define feature groups
        self.binary_features = [
            'gender', 'partner', 'dependents', 'phone_service',
            'paperless_billing'
        ]
        
        self.categorical_features = [
            'multiple_lines', 'internet_service', 'online_security',
            'online_backup', 'device_protection', 'tech_support',
            'streaming_tv', 'streaming_movies', 'contract', 'payment_method'
        ]
        
        self.numerical_features = [
            'tenure', 'monthly_charges', 'total_charges'
        ]
        
        logger.info("FeatureTransformer initialized")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'churn') -> 'FeatureTransformer':
        """Fit the transformer on training data.
        
        Args:
            df: Training DataFrame
            target_col: Name of the target column
            
        Returns:
            self for method chaining
        """
        logger.info("Fitting FeatureTransformer on training data")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Fit label encoders for binary features
        for col in self.binary_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Fitted LabelEncoder for {col}")
        
        # Fit scaler on numerical features
        numerical_data = df[self.numerical_features].copy()
        self.scaler.fit(numerical_data)
        logger.debug(f"Fitted StandardScaler on {len(self.numerical_features)} numerical features")
        
        self.is_fitted = True
        logger.info("FeatureTransformer fitting complete")
        
        return self
    
    def transform(self, df: pd.DataFrame, target_col: str = 'churn') -> pd.DataFrame:
        """Transform data using fitted transformers.
        
        Args:
            df: DataFrame to transform
            target_col: Name of the target column
            
        Returns:
            Transformed DataFrame
            
        Raises:
            ValueError: If transformer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("FeatureTransformer must be fitted before transform")
        
        logger.info(f"Transforming data with {len(df)} records")
        
        # Create a copy
        df = df.copy()
        
        # 1. Encode binary features
        for col in self.binary_features:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
                logger.debug(f"Encoded binary feature: {col}")
        
        # 2. One-hot encode categorical features
        df = pd.get_dummies(df, columns=self.categorical_features, drop_first=True)
        logger.debug(f"One-hot encoded {len(self.categorical_features)} categorical features")
        
        # 3. Create derived features
        df = self._create_derived_features(df)
        
        # 4. Scale numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            logger.debug(f"Scaled {len(numerical_cols)} numerical features")
        
        # 5. Encode target variable if present
        if target_col in df.columns:
            df[target_col] = (df[target_col] == 'Yes').astype(int)
            logger.debug(f"Encoded target variable: {target_col}")
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != target_col and col != 'customer_id']
        
        logger.info(f"Transformation complete. Output shape: {df.shape}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'churn') -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            target_col: Name of the target column
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df, target_col).transform(df, target_col)
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with additional derived features
        """
        logger.debug("Creating derived features")
        
        # 1. Charges per month (average monthly charge over tenure)
        # Avoid division by zero
        df['charges_per_month'] = df['total_charges'] / (df['tenure'] + 1)
        
        # 2. Service count (number of additional services)
        service_cols = [
            'phone_service', 'multiple_lines', 'internet_service',
            'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        # Count services (excluding 'No' and 'No internet service' values)
        df['service_count'] = 0
        for col in service_cols:
            if col in df.columns:
                # For binary encoded columns, just add the value
                if col in self.binary_features:
                    df['service_count'] += df[col]
                # For categorical columns, check if not 'No' or 'No internet service'
                else:
                    df['service_count'] += (~df[col].astype(str).isin(['No', 'No internet service', 'No phone service'])).astype(int)
        
        # 3. Tenure group (categorize tenure into groups)
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, 72],
            labels=[0, 1, 2, 3]  # 0-12, 13-24, 25-48, 49-72 months
        ).astype(int)
        
        # 4. Is senior with dependents
        if 'senior_citizen' in df.columns and 'dependents' in df.columns:
            df['senior_with_dependents'] = (df['senior_citizen'] == 1) & (df['dependents'] == 1)
            df['senior_with_dependents'] = df['senior_with_dependents'].astype(int)
        
        # 5. High value customer (high monthly charges and long tenure)
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            df['high_value_customer'] = (
                (df['monthly_charges'] > df['monthly_charges'].median()) &
                (df['tenure'] > df['tenure'].median())
            ).astype(int)
        
        logger.debug(f"Created 5 derived features")
        
        return df
    
    def save(self, filepath: str) -> None:
        """Save the fitted transformer to disk.
        
        Args:
            filepath: Path to save the transformer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformer")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save transformer state
        state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'binary_features': self.binary_features,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }
        
        joblib.dump(state, filepath)
        logger.info(f"FeatureTransformer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureTransformer':
        """Load a fitted transformer from disk.
        
        Args:
            filepath: Path to the saved transformer
            
        Returns:
            Loaded FeatureTransformer instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Transformer file not found: {filepath}")
        
        state = joblib.load(filepath)
        
        # Create new instance and restore state
        transformer = cls()
        transformer.scaler = state['scaler']
        transformer.label_encoders = state['label_encoders']
        transformer.feature_names = state['feature_names']
        transformer.is_fitted = state['is_fitted']
        transformer.binary_features = state['binary_features']
        transformer.categorical_features = state['categorical_features']
        transformer.numerical_features = state['numerical_features']
        
        logger.info(f"FeatureTransformer loaded from {filepath}")
        
        return transformer
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted first")
        
        return self.feature_names.copy()


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'churn',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train-test split with optional stratification.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split by target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Preparing train-test split (test_size={test_size}, stratify={stratify})")
    
    # Separate features and target
    X = df.drop(columns=[target_col, 'customer_id'], errors='ignore')
    y = df[target_col]
    
    # Perform split
    stratify_col = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    logger.info(f"Split complete: train={len(X_train)}, test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test
