"""Unit tests for FeatureTransformer class.

Comprehensive test suite covering feature encoding, scaling,
derived feature creation, and transformer persistence.

Author: Jay Rathod
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.features import FeatureTransformer, prepare_train_test_split


class TestFeatureTransformer:
    """Test suite for FeatureTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample customer churn data for testing."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'senior_citizen': [0, 1, 0, 0, 1],
            'partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [12, 24, 6, 36, 48],
            'phone_service': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'multiple_lines': ['No', 'Yes', 'No phone service', 'Yes', 'No'],
            'internet_service': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
            'online_security': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'online_backup': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'device_protection': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'tech_support': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'streaming_tv': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'streaming_movies': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
            'paperless_billing': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
                             'Credit card (automatic)', 'Electronic check'],
            'monthly_charges': [50.0, 75.5, 30.25, 85.0, 95.5],
            'total_charges': [600.0, 1812.0, 181.5, 3060.0, 4584.0],
            'churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })
    
    @pytest.fixture
    def transformer(self):
        """Create a FeatureTransformer instance."""
        return FeatureTransformer()
    
    def test_initialization(self, transformer):
        """Test FeatureTransformer initialization."""
        assert transformer.is_fitted is False
        assert len(transformer.label_encoders) == 0
        assert len(transformer.feature_names) == 0
        assert transformer.scaler is not None
    
    def test_fit(self, transformer, sample_data):
        """Test fitting the transformer."""
        transformer.fit(sample_data)
        
        assert transformer.is_fitted is True
        assert len(transformer.label_encoders) > 0
        assert transformer.scaler is not None
    
    def test_transform_without_fit_raises_error(self, transformer, sample_data):
        """Test that transform raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            transformer.transform(sample_data)
    
    def test_fit_transform(self, transformer, sample_data):
        """Test fit_transform method."""
        result = transformer.fit_transform(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert transformer.is_fitted is True
    
    def test_binary_encoding(self, transformer, sample_data):
        """Test binary feature encoding."""
        result = transformer.fit_transform(sample_data)
        
        # Check that binary features are encoded as 0/1
        for col in transformer.binary_features:
            if col in result.columns:
                assert result[col].isin([0, 1]).all()
    
    def test_categorical_encoding(self, transformer, sample_data):
        """Test one-hot encoding of categorical features."""
        result = transformer.fit_transform(sample_data)
        
        # Check that original categorical columns are removed
        for col in transformer.categorical_features:
            assert col not in result.columns
        
        # Check that one-hot encoded columns are created
        # (should have columns like 'internet_service_Fiber optic', etc.)
        encoded_cols = [col for col in result.columns if any(cat in col for cat in transformer.categorical_features)]
        assert len(encoded_cols) > 0
    
    def test_numerical_scaling(self, transformer, sample_data):
        """Test numerical feature scaling."""
        result = transformer.fit_transform(sample_data)
        
        # Check that numerical features are scaled (mean ~0, std ~1)
        for col in transformer.numerical_features:
            if col in result.columns:
                assert abs(result[col].mean()) < 1.0  # Should be close to 0
                assert abs(result[col].std() - 1.0) < 0.5  # Should be close to 1
    
    def test_derived_features_created(self, transformer, sample_data):
        """Test that derived features are created."""
        result = transformer.fit_transform(sample_data)
        
        # Check for derived features
        assert 'charges_per_month' in result.columns
        assert 'service_count' in result.columns
        assert 'tenure_group' in result.columns
        assert 'senior_with_dependents' in result.columns
        assert 'high_value_customer' in result.columns
    
    def test_charges_per_month_calculation(self, transformer, sample_data):
        """Test charges_per_month derived feature."""
        result = transformer.fit_transform(sample_data)
        
        # Verify calculation (before scaling)
        # charges_per_month = total_charges / (tenure + 1)
        assert 'charges_per_month' in result.columns
        assert not result['charges_per_month'].isnull().any()
    
    def test_service_count_calculation(self, transformer, sample_data):
        """Test service_count derived feature."""
        result = transformer.fit_transform(sample_data)
        
        assert 'service_count' in result.columns
        assert result['service_count'].min() >= 0
        assert result['service_count'].max() <= 10  # Max possible services
    
    def test_tenure_group_creation(self, transformer, sample_data):
        """Test tenure_group derived feature."""
        result = transformer.fit_transform(sample_data)
        
        assert 'tenure_group' in result.columns
        assert result['tenure_group'].isin([0, 1, 2, 3]).all()
    
    def test_target_encoding(self, transformer, sample_data):
        """Test target variable encoding."""
        result = transformer.fit_transform(sample_data)
        
        # Target should be encoded as 0/1
        assert result['churn'].isin([0, 1]).all()
        # 'Yes' should be 1, 'No' should be 0
        assert (result['churn'] == 1).sum() == (sample_data['churn'] == 'Yes').sum()
    
    def test_customer_id_preserved(self, transformer, sample_data):
        """Test that customer_id is preserved."""
        result = transformer.fit_transform(sample_data)
        
        if 'customer_id' in result.columns:
            assert result['customer_id'].tolist() == sample_data['customer_id'].tolist()
    
    def test_get_feature_names(self, transformer, sample_data):
        """Test getting feature names after transformation."""
        transformer.fit_transform(sample_data)
        
        feature_names = transformer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'churn' not in feature_names
        assert 'customer_id' not in feature_names
    
    def test_get_feature_names_before_fit_raises_error(self, transformer):
        """Test that get_feature_names raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            transformer.get_feature_names()
    
    def test_save_and_load(self, transformer, sample_data):
        """Test saving and loading transformer."""
        # Fit transformer
        transformer.fit_transform(sample_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_file = f.name
        
        try:
            transformer.save(temp_file)
            
            # Load transformer
            loaded_transformer = FeatureTransformer.load(temp_file)
            
            # Verify loaded transformer
            assert loaded_transformer.is_fitted is True
            assert len(loaded_transformer.label_encoders) == len(transformer.label_encoders)
            assert loaded_transformer.feature_names == transformer.feature_names
            
            # Test that loaded transformer can transform data
            result = loaded_transformer.transform(sample_data)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_data)
        
        finally:
            Path(temp_file).unlink()
    
    def test_save_unfitted_raises_error(self, transformer):
        """Test that saving unfitted transformer raises error."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Cannot save unfitted"):
                transformer.save(temp_file)
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureTransformer.load('nonexistent_file.joblib')
    
    def test_transform_consistency(self, transformer, sample_data):
        """Test that transform produces consistent results."""
        # Fit once
        transformer.fit(sample_data)
        
        # Transform twice
        result1 = transformer.transform(sample_data)
        result2 = transformer.transform(sample_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestPrepareTrainTestSplit:
    """Test suite for prepare_train_test_split function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'customer_id': [f'C{i:03d}' for i in range(100)],
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'churn': np.random.choice([0, 1], 100)
        })
    
    def test_basic_split(self, sample_data):
        """Test basic train-test split."""
        X_train, X_test, y_train, y_test = prepare_train_test_split(sample_data)
        
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_test_size(self, sample_data):
        """Test that test_size parameter works correctly."""
        test_size = 0.3
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            sample_data, test_size=test_size
        )
        
        expected_test_size = int(len(sample_data) * test_size)
        assert abs(len(X_test) - expected_test_size) <= 1  # Allow for rounding
    
    def test_stratification(self, sample_data):
        """Test stratified split maintains class distribution."""
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            sample_data, stratify=True
        )
        
        # Check that class distribution is similar in train and test
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        overall_ratio = sample_data['churn'].mean()
        
        assert abs(train_ratio - overall_ratio) < 0.1
        assert abs(test_ratio - overall_ratio) < 0.1
    
    def test_random_state(self, sample_data):
        """Test that random_state produces reproducible splits."""
        X_train1, X_test1, y_train1, y_test1 = prepare_train_test_split(
            sample_data, random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = prepare_train_test_split(
            sample_data, random_state=42
        )
        
        # Splits should be identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
    
    def test_customer_id_excluded(self, sample_data):
        """Test that customer_id is excluded from features."""
        X_train, X_test, y_train, y_test = prepare_train_test_split(sample_data)
        
        assert 'customer_id' not in X_train.columns
        assert 'customer_id' not in X_test.columns
    
    def test_target_excluded_from_features(self, sample_data):
        """Test that target is excluded from features."""
        X_train, X_test, y_train, y_test = prepare_train_test_split(sample_data)
        
        assert 'churn' not in X_train.columns
        assert 'churn' not in X_test.columns
