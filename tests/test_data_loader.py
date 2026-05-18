"""Unit tests for DataLoader class."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.data import DataLoader, ValidationResult
from src.utils.exceptions import DataLoadError


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample customer churn data."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'gender': ['Male', 'Female', 'Male'],
            'senior_citizen': [0, 1, 0],
            'partner': ['Yes', 'No', 'Yes'],
            'dependents': ['No', 'Yes', 'No'],
            'tenure': [12, 24, 6],
            'phone_service': ['Yes', 'Yes', 'No'],
            'multiple_lines': ['No', 'Yes', 'No phone service'],
            'internet_service': ['DSL', 'Fiber optic', 'No'],
            'online_security': ['Yes', 'No', 'No internet service'],
            'online_backup': ['No', 'Yes', 'No internet service'],
            'device_protection': ['Yes', 'No', 'No internet service'],
            'tech_support': ['No', 'Yes', 'No internet service'],
            'streaming_tv': ['Yes', 'No', 'No internet service'],
            'streaming_movies': ['No', 'Yes', 'No internet service'],
            'contract': ['Month-to-month', 'One year', 'Two year'],
            'paperless_billing': ['Yes', 'No', 'Yes'],
            'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
            'monthly_charges': [50.0, 75.5, 30.25],
            'total_charges': [600.0, 1812.0, 181.5],
            'churn': ['No', 'Yes', 'No']
        })
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance."""
        return DataLoader()
    
    def test_load_data_success(self, data_loader, sample_data):
        """Test successful data loading from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            df = data_loader.load_data(temp_file)
            assert len(df) == 3
            assert len(df.columns) == 21
            assert 'customer_id' in df.columns
        finally:
            Path(temp_file).unlink()
    
    def test_load_data_file_not_found(self, data_loader):
        """Test loading data from non-existent file."""
        with pytest.raises(DataLoadError) as exc_info:
            data_loader.load_data('nonexistent_file.csv')
        
        assert 'not found' in str(exc_info.value).lower()
    
    def test_load_data_empty_file(self, data_loader):
        """Test loading data from empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')
            temp_file = f.name
        
        try:
            with pytest.raises(DataLoadError) as exc_info:
                data_loader.load_data(temp_file)
            
            assert 'empty' in str(exc_info.value).lower()
        finally:
            Path(temp_file).unlink()
    
    def test_validate_schema_success(self, data_loader, sample_data):
        """Test successful schema validation."""
        result = data_loader.validate_schema(sample_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_schema_missing_columns(self, data_loader, sample_data):
        """Test schema validation with missing required columns."""
        # Remove some required columns
        df_missing = sample_data.drop(columns=['tenure', 'monthly_charges'])
        
        result = data_loader.validate_schema(df_missing)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('tenure' in error for error in result.errors)
        assert any('monthly_charges' in error for error in result.errors)
    
    def test_validate_schema_wrong_dtype(self, data_loader, sample_data):
        """Test schema validation with incorrect data types."""
        # Change data type of a column
        df_wrong_type = sample_data.copy()
        df_wrong_type['tenure'] = df_wrong_type['tenure'].astype(str)
        
        result = data_loader.validate_schema(df_wrong_type)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('tenure' in error and 'data type' in error.lower() for error in result.errors)
    
    def test_validate_schema_extra_columns(self, data_loader, sample_data):
        """Test schema validation with extra columns (should be warning only)."""
        # Add extra column
        df_extra = sample_data.copy()
        df_extra['extra_column'] = [1, 2, 3]
        
        result = data_loader.validate_schema(df_extra)
        
        # Extra columns should not cause validation to fail
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any('extra' in warning.lower() for warning in result.warnings)
    
    def test_validate_schema_empty_dataframe(self, data_loader):
        """Test schema validation with empty DataFrame."""
        df_empty = pd.DataFrame()
        
        result = data_loader.validate_schema(df_empty)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('empty' in error.lower() for error in result.errors)
    
    def test_dtypes_compatible(self):
        """Test data type compatibility checking."""
        # Exact match
        assert DataLoader._dtypes_compatible('int64', 'int64') is True
        
        # Integer compatibility
        assert DataLoader._dtypes_compatible('int32', 'int64') is True
        assert DataLoader._dtypes_compatible('int64', 'int32') is True
        
        # Float compatibility
        assert DataLoader._dtypes_compatible('float32', 'float64') is True
        assert DataLoader._dtypes_compatible('float64', 'float32') is True
        
        # Object/string compatibility
        assert DataLoader._dtypes_compatible('object', 'object') is True
        assert DataLoader._dtypes_compatible('string', 'object') is True
        
        # Incompatible types
        assert DataLoader._dtypes_compatible('int64', 'float64') is False
        assert DataLoader._dtypes_compatible('object', 'int64') is False


class TestMissingValueHandling:
    """Test suite for missing value handling functionality."""
    
    @pytest.fixture
    def data_with_missing(self):
        """Create sample data with missing values."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25.0, 30.0, np.nan, 40.0, 35.0],
            'income': [50000.0, np.nan, 60000.0, 70000.0, np.nan],
            'tenure': [12, 24, np.nan, 36, 6],
            'category': ['A', 'B', np.nan, 'A', 'C'],
            'status': ['Active', np.nan, 'Active', 'Inactive', 'Active']
        })
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance."""
        return DataLoader()
    
    def test_impute_mean(self, data_loader, data_with_missing):
        """Test mean imputation for numerical columns."""
        result = data_loader._impute_mean(data_with_missing, ['age', 'income'])
        
        # Check that no missing values remain in imputed columns
        assert result['age'].isnull().sum() == 0
        assert result['income'].isnull().sum() == 0
        
        # Check that mean was calculated correctly
        # age: (25 + 30 + 40 + 35) / 4 = 32.5
        expected_age_mean = 32.5
        assert result.loc[2, 'age'] == expected_age_mean
        
        # income: (50000 + 60000 + 70000) / 3 = 60000
        expected_income_mean = 60000.0
        assert result.loc[1, 'income'] == expected_income_mean
        assert result.loc[4, 'income'] == expected_income_mean
    
    def test_impute_median(self, data_loader, data_with_missing):
        """Test median imputation for numerical columns."""
        result = data_loader._impute_median(data_with_missing, ['age', 'income'])
        
        # Check that no missing values remain in imputed columns
        assert result['age'].isnull().sum() == 0
        assert result['income'].isnull().sum() == 0
        
        # Check that median was calculated correctly
        # age: median of [25, 30, 35, 40] = 32.5
        expected_age_median = 32.5
        assert result.loc[2, 'age'] == expected_age_median
        
        # income: median of [50000, 60000, 70000] = 60000
        expected_income_median = 60000.0
        assert result.loc[1, 'income'] == expected_income_median
    
    def test_impute_mode(self, data_loader, data_with_missing):
        """Test mode imputation for categorical columns."""
        result = data_loader._impute_mode(data_with_missing, ['category', 'status'])
        
        # Check that no missing values remain in imputed columns
        assert result['category'].isnull().sum() == 0
        assert result['status'].isnull().sum() == 0
        
        # Check that mode was used correctly
        # category: mode is 'A' (appears twice)
        assert result.loc[2, 'category'] == 'A'
        
        # status: mode is 'Active' (appears 3 times)
        assert result.loc[1, 'status'] == 'Active'
    
    def test_impute_constant(self, data_loader, data_with_missing):
        """Test constant imputation for categorical columns."""
        result = data_loader._impute_constant(data_with_missing, ['category', 'status'], fill_value='Unknown')
        
        # Check that no missing values remain in imputed columns
        assert result['category'].isnull().sum() == 0
        assert result['status'].isnull().sum() == 0
        
        # Check that constant value was used
        assert result.loc[2, 'category'] == 'Unknown'
        assert result.loc[1, 'status'] == 'Unknown'
    
    def test_drop_missing_rows(self, data_loader, data_with_missing):
        """Test dropping rows with missing values."""
        result = data_loader._drop_missing(data_with_missing, ['age', 'income'])
        
        # Should drop rows with missing values in age or income
        # Row 1 has missing income, row 2 has missing age, row 4 has missing income
        # Should keep rows 0 and 3 only
        assert len(result) == 2
        assert result['customer_id'].tolist() == ['C001', 'C004']
        
        # Check that remaining rows have no missing values in specified columns
        assert result['age'].isnull().sum() == 0
        assert result['income'].isnull().sum() == 0
    
    def test_drop_columns_with_missing(self, data_loader):
        """Test dropping columns with high proportion of missing values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            'col3': [1, 2, np.nan, 4, 5],  # 20% missing
            'col4': [np.nan, np.nan, np.nan, np.nan, np.nan]  # 100% missing
        })
        
        # Drop columns with >50% missing values
        result = data_loader.drop_columns_with_missing(df, threshold=0.5)
        
        # col2 (80% missing) and col4 (100% missing) should be dropped
        assert 'col1' in result.columns
        assert 'col2' not in result.columns
        assert 'col3' in result.columns
        assert 'col4' not in result.columns
    
    def test_handle_missing_values_mean_strategy(self, data_loader, data_with_missing):
        """Test handle_missing_values with mean strategy for numerical columns."""
        strategy = {'numerical': 'mean', 'categorical': 'mode'}
        
        result = data_loader.handle_missing_values(
            data_with_missing, 
            strategy,
            numerical_columns=['age', 'income', 'tenure'],
            categorical_columns=['category', 'status']
        )
        
        # Check that no missing values remain
        assert result['age'].isnull().sum() == 0
        assert result['income'].isnull().sum() == 0
        assert result['tenure'].isnull().sum() == 0
        assert result['category'].isnull().sum() == 0
        assert result['status'].isnull().sum() == 0
    
    def test_handle_missing_values_median_strategy(self, data_loader, data_with_missing):
        """Test handle_missing_values with median strategy for numerical columns."""
        strategy = {'numerical': 'median', 'categorical': 'mode'}
        
        result = data_loader.handle_missing_values(
            data_with_missing, 
            strategy,
            numerical_columns=['age', 'income'],
            categorical_columns=['category', 'status']
        )
        
        # Check that no missing values remain
        assert result['age'].isnull().sum() == 0
        assert result['income'].isnull().sum() == 0
        assert result['category'].isnull().sum() == 0
        assert result['status'].isnull().sum() == 0
    
    def test_handle_missing_values_drop_strategy(self, data_loader, data_with_missing):
        """Test handle_missing_values with drop strategy."""
        strategy = {'numerical': 'drop', 'categorical': 'drop'}
        
        result = data_loader.handle_missing_values(
            data_with_missing, 
            strategy,
            numerical_columns=['age', 'income'],
            categorical_columns=['category', 'status']
        )
        
        # Should only keep rows with no missing values in any specified column
        # Only row 0 and row 3 have no missing values in the specified columns
        # But row 3 has missing tenure, so if we include tenure, only row 0 remains
        # Let's check without tenure
        assert len(result) <= len(data_with_missing)
        
        # Check that no missing values remain in specified columns
        assert result['age'].isnull().sum() == 0
        assert result['income'].isnull().sum() == 0
        assert result['category'].isnull().sum() == 0
        assert result['status'].isnull().sum() == 0
    
    def test_handle_missing_values_auto_detect_columns(self, data_loader, data_with_missing):
        """Test handle_missing_values with auto-detection of column types."""
        strategy = {'numerical': 'median', 'categorical': 'mode'}
        
        # Don't specify columns, let it auto-detect
        result = data_loader.handle_missing_values(data_with_missing, strategy)
        
        # Check that missing values were handled
        # Numerical columns should be imputed with median
        # Categorical columns should be imputed with mode
        assert result.isnull().sum().sum() == 0  # No missing values should remain
    
    def test_handle_missing_values_invalid_strategy(self, data_loader, data_with_missing):
        """Test handle_missing_values with invalid strategy."""
        from src.utils.exceptions import DataValidationError
        
        strategy = {'numerical': 'invalid_strategy', 'categorical': 'mode'}
        
        with pytest.raises(DataValidationError) as exc_info:
            data_loader.handle_missing_values(data_with_missing, strategy)
        
        assert 'invalid' in str(exc_info.value).lower()
    
    def test_handle_missing_values_preserves_non_missing_data(self, data_loader, data_with_missing):
        """Test that handle_missing_values preserves non-missing data."""
        strategy = {'numerical': 'mean', 'categorical': 'mode'}
        
        result = data_loader.handle_missing_values(
            data_with_missing, 
            strategy,
            numerical_columns=['age', 'income'],
            categorical_columns=['category', 'status']
        )
        
        # Check that non-missing values are preserved
        assert result.loc[0, 'age'] == 25.0
        assert result.loc[1, 'age'] == 30.0
        assert result.loc[0, 'income'] == 50000.0
        assert result.loc[0, 'category'] == 'A'
        assert result.loc[0, 'status'] == 'Active'
    
    def test_impute_mode_with_no_mode(self, data_loader):
        """Test mode imputation when column has no mode (all NaN)."""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan]
        })
        
        result = data_loader._impute_mode(df, ['col1'])
        
        # Should still have missing values since no mode exists
        assert result['col1'].isnull().all()
    
    def test_drop_missing_with_nonexistent_columns(self, data_loader, data_with_missing):
        """Test drop_missing with columns that don't exist in DataFrame."""
        result = data_loader._drop_missing(data_with_missing, ['nonexistent_col'])
        
        # Should return DataFrame unchanged
        assert len(result) == len(data_with_missing)
    
    def test_handle_missing_values_constant_strategy(self, data_loader, data_with_missing):
        """Test handle_missing_values with constant strategy for categorical columns."""
        strategy = {'numerical': 'median', 'categorical': 'constant'}
        
        result = data_loader.handle_missing_values(
            data_with_missing, 
            strategy,
            numerical_columns=['age', 'income'],
            categorical_columns=['category', 'status']
        )
        
        # Check that missing categorical values were filled with 'Unknown'
        assert result.loc[2, 'category'] == 'Unknown'
        assert result.loc[1, 'status'] == 'Unknown'
        
        # Check that no missing values remain
        assert result.isnull().sum().sum() == 0
