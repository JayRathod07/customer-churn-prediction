"""Data loading and validation module for customer churn prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
import logging

from ..utils.exceptions import DataLoadError, DataValidationError

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of schema validation."""
    
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether validation passed
            errors: List of validation errors
            warnings: List of validation warnings
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self.is_valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"


class DataLoader:
    """Loads and validates customer churn data from CSV files."""
    
    def __init__(self, required_columns: Optional[List[str]] = None, expected_dtypes: Optional[Dict[str, str]] = None):
        """
        Initialize the DataLoader.
        
        Args:
            required_columns: List of required column names
            expected_dtypes: Dictionary mapping column names to expected data types
        """
        self.required_columns = required_columns or self._get_default_required_columns()
        self.expected_dtypes = expected_dtypes or self._get_default_expected_dtypes()
        logger.info(f"DataLoader initialized with {len(self.required_columns)} required columns")
    
    @staticmethod
    def _get_default_required_columns() -> List[str]:
        """
        Get default list of required columns for customer churn data.
        
        Returns:
            List of required column names
        """
        return [
            'customer_id',
            'gender',
            'senior_citizen',
            'partner',
            'dependents',
            'tenure',
            'phone_service',
            'multiple_lines',
            'internet_service',
            'online_security',
            'online_backup',
            'device_protection',
            'tech_support',
            'streaming_tv',
            'streaming_movies',
            'contract',
            'paperless_billing',
            'payment_method',
            'monthly_charges',
            'total_charges'
        ]
    
    @staticmethod
    def _get_default_expected_dtypes() -> Dict[str, str]:
        """
        Get default expected data types for customer churn data columns.
        
        Returns:
            Dictionary mapping column names to expected data types
        """
        return {
            'customer_id': 'object',
            'gender': 'object',
            'senior_citizen': 'int64',
            'partner': 'object',
            'dependents': 'object',
            'tenure': 'int64',
            'phone_service': 'object',
            'multiple_lines': 'object',
            'internet_service': 'object',
            'online_security': 'object',
            'online_backup': 'object',
            'device_protection': 'object',
            'tech_support': 'object',
            'streaming_tv': 'object',
            'streaming_movies': 'object',
            'contract': 'object',
            'paperless_billing': 'object',
            'payment_method': 'object',
            'monthly_charges': 'float64',
            'total_charges': 'float64',
            'churn': 'object'
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pandas DataFrame containing the loaded data
            
        Raises:
            DataLoadError: If the file cannot be loaded
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            error_msg = f"CSV file not found: {file_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        # Check if file is readable
        if not file_path.is_file():
            error_msg = f"Path is not a file: {file_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        try:
            # Load CSV file
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Check if DataFrame is empty
            if df.empty:
                error_msg = f"CSV file is empty: {file_path}"
                logger.error(error_msg)
                raise DataLoadError(error_msg)
            
            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            return df
            
        except pd.errors.EmptyDataError as e:
            error_msg = f"CSV file is empty or has no data: {file_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e
        
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file: {file_path}. Error: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Unexpected error loading CSV file: {file_path}. Error: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg) from e
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate DataFrame schema against expected columns and data types.
        
        Args:
            df: pandas DataFrame to validate
            
        Returns:
            ValidationResult object containing validation status and any errors/warnings
            
        Raises:
            DataValidationError: If validation fails critically
        """
        errors = []
        warnings = []
        
        logger.info("Validating DataFrame schema")
        
        # Check for missing required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            error_msg = f"Missing required columns: {sorted(missing_columns)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Check for extra columns (warning only)
        extra_columns = set(df.columns) - set(self.expected_dtypes.keys())
        if extra_columns:
            warning_msg = f"Extra columns found (not in schema): {sorted(extra_columns)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Validate data types for columns that exist
        for col in df.columns:
            if col in self.expected_dtypes:
                expected_dtype = self.expected_dtypes[col]
                actual_dtype = str(df[col].dtype)
                
                # Check if data type matches (with some flexibility for numeric types)
                if not self._dtypes_compatible(actual_dtype, expected_dtype):
                    error_msg = f"Column '{col}' has incorrect data type. Expected: {expected_dtype}, Got: {actual_dtype}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        # Check for completely empty DataFrame
        if df.empty:
            error_msg = "DataFrame is empty (no rows)"
            errors.append(error_msg)
            logger.error(error_msg)
        
        # Determine if validation passed
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.error(f"Schema validation failed with {len(errors)} errors")
        
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    @staticmethod
    def _dtypes_compatible(actual: str, expected: str) -> bool:
        """
        Check if actual and expected data types are compatible.
        
        Allows some flexibility for numeric types (e.g., int32 vs int64, float32 vs float64).
        
        Args:
            actual: Actual data type as string
            expected: Expected data type as string
            
        Returns:
            True if types are compatible, False otherwise
        """
        # Exact match
        if actual == expected:
            return True
        
        # Integer type compatibility
        if expected in ['int64', 'int32', 'int16', 'int8'] and actual in ['int64', 'int32', 'int16', 'int8']:
            return True
        
        # Float type compatibility
        if expected in ['float64', 'float32'] and actual in ['float64', 'float32']:
            return True
        
        # Object/string compatibility (pandas uses 'object' for strings, but newer versions may use 'str' or 'string')
        if expected == 'object' and actual in ['object', 'string', 'str']:
            return True
        
        if expected == 'string' and actual in ['object', 'string', 'str']:
            return True
        
        if expected == 'str' and actual in ['object', 'string', 'str']:
            return True
        
        return False
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: Dict[str, str],
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame according to the specified strategy.
        
        Args:
            df: pandas DataFrame with potential missing values
            strategy: Dictionary with 'numerical' and 'categorical' keys specifying strategies
                     Options for numerical: 'mean', 'median', 'drop'
                     Options for categorical: 'mode', 'drop', 'constant'
            numerical_columns: List of numerical column names (auto-detected if None)
            categorical_columns: List of categorical column names (auto-detected if None)
            
        Returns:
            pandas DataFrame with missing values handled
            
        Raises:
            DataValidationError: If strategy is invalid
        """
        logger.info("Handling missing values")
        df = df.copy()
        
        # Auto-detect column types if not provided
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Get strategies
        numerical_strategy = strategy.get('numerical', 'median')
        categorical_strategy = strategy.get('categorical', 'mode')
        
        # Validate strategies
        valid_numerical_strategies = ['mean', 'median', 'drop']
        valid_categorical_strategies = ['mode', 'drop', 'constant']
        
        if numerical_strategy not in valid_numerical_strategies:
            raise DataValidationError(
                f"Invalid numerical strategy: {numerical_strategy}. "
                f"Valid options: {valid_numerical_strategies}"
            )
        
        if categorical_strategy not in valid_categorical_strategies:
            raise DataValidationError(
                f"Invalid categorical strategy: {categorical_strategy}. "
                f"Valid options: {valid_categorical_strategies}"
            )
        
        # Handle numerical columns
        if numerical_strategy == 'mean':
            df = self._impute_mean(df, numerical_columns)
        elif numerical_strategy == 'median':
            df = self._impute_median(df, numerical_columns)
        elif numerical_strategy == 'drop':
            df = self._drop_missing(df, numerical_columns)
        
        # Handle categorical columns
        if categorical_strategy == 'mode':
            df = self._impute_mode(df, categorical_columns)
        elif categorical_strategy == 'constant':
            df = self._impute_constant(df, categorical_columns, fill_value='Unknown')
        elif categorical_strategy == 'drop':
            df = self._drop_missing(df, categorical_columns)
        
        logger.info(f"Missing value handling complete. Remaining rows: {len(df)}")
        return df
    
    @staticmethod
    def _impute_mean(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Impute missing values with the mean of each column.
        
        Args:
            df: pandas DataFrame
            columns: List of column names to impute
            
        Returns:
            DataFrame with mean imputation applied
        """
        df = df.copy()
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                logger.debug(f"Imputed {col} with mean: {mean_value}")
        return df
    
    @staticmethod
    def _impute_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Impute missing values with the median of each column.
        
        Args:
            df: pandas DataFrame
            columns: List of column names to impute
            
        Returns:
            DataFrame with median imputation applied
        """
        df = df.copy()
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.debug(f"Imputed {col} with median: {median_value}")
        return df
    
    @staticmethod
    def _impute_mode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Impute missing values with the mode (most frequent value) of each column.
        
        Args:
            df: pandas DataFrame
            columns: List of column names to impute
            
        Returns:
            DataFrame with mode imputation applied
        """
        df = df.copy()
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                mode_values = df[col].mode()
                if len(mode_values) > 0:
                    mode_value = mode_values[0]
                    df[col].fillna(mode_value, inplace=True)
                    logger.debug(f"Imputed {col} with mode: {mode_value}")
                else:
                    logger.warning(f"No mode found for column {col}, skipping imputation")
        return df
    
    @staticmethod
    def _impute_constant(df: pd.DataFrame, columns: List[str], fill_value: Any = 'Unknown') -> pd.DataFrame:
        """
        Impute missing values with a constant value.
        
        Args:
            df: pandas DataFrame
            columns: List of column names to impute
            fill_value: Constant value to use for imputation
            
        Returns:
            DataFrame with constant imputation applied
        """
        df = df.copy()
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(fill_value, inplace=True)
                logger.debug(f"Imputed {col} with constant: {fill_value}")
        return df
    
    @staticmethod
    def _drop_missing(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop rows with missing values in the specified columns.
        
        Args:
            df: pandas DataFrame
            columns: List of column names to check for missing values
            
        Returns:
            DataFrame with rows containing missing values removed
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Only consider columns that exist in the DataFrame
        existing_columns = [col for col in columns if col in df.columns]
        
        if existing_columns:
            df = df.dropna(subset=existing_columns)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                logger.debug(f"Dropped {dropped_rows} rows with missing values in {existing_columns}")
        
        return df
    
    @staticmethod
    def drop_columns_with_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Drop columns with missing values exceeding the threshold.
        
        Args:
            df: pandas DataFrame
            threshold: Proportion of missing values (0.0 to 1.0) above which to drop the column
            
        Returns:
            DataFrame with high-missing-value columns removed
        """
        df = df.copy()
        initial_columns = len(df.columns)
        
        # Calculate missing value proportion for each column
        missing_proportions = df.isnull().sum() / len(df)
        
        # Identify columns to drop
        columns_to_drop = missing_proportions[missing_proportions > threshold].index.tolist()
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Dropped {len(columns_to_drop)} columns with >{threshold*100}% missing values: {columns_to_drop}")
        
        return df


class DataQualityReport:
    """Data quality assessment report."""
    
    def __init__(
        self,
        total_records: int,
        total_features: int,
        missing_values: Dict[str, int],
        missing_percentage: Dict[str, float],
        data_types: Dict[str, str],
        numerical_stats: Dict[str, Dict[str, float]],
        categorical_stats: Dict[str, Dict[str, int]],
        duplicates_count: int,
        validation_errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize data quality report.
        
        Args:
            total_records: Total number of records in the dataset
            total_features: Total number of features/columns
            missing_values: Dictionary mapping column names to count of missing values
            missing_percentage: Dictionary mapping column names to percentage of missing values
            data_types: Dictionary mapping column names to data types
            numerical_stats: Dictionary mapping numerical column names to statistics (mean, std, min, max, etc.)
            categorical_stats: Dictionary mapping categorical column names to value counts
            duplicates_count: Number of duplicate records
            validation_errors: List of validation errors found
            warnings: List of warnings
            timestamp: Timestamp when report was generated
        """
        self.total_records = total_records
        self.total_features = total_features
        self.missing_values = missing_values
        self.missing_percentage = missing_percentage
        self.data_types = data_types
        self.numerical_stats = numerical_stats
        self.categorical_stats = categorical_stats
        self.duplicates_count = duplicates_count
        self.validation_errors = validation_errors or []
        self.warnings = warnings or []
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self) -> str:
        return (
            f"DataQualityReport(records={self.total_records}, "
            f"features={self.total_features}, "
            f"duplicates={self.duplicates_count}, "
            f"errors={len(self.validation_errors)})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary format.
        
        Returns:
            Dictionary representation of the report
        """
        return {
            'total_records': self.total_records,
            'total_features': self.total_features,
            'missing_values': self.missing_values,
            'missing_percentage': self.missing_percentage,
            'data_types': self.data_types,
            'numerical_stats': self.numerical_stats,
            'categorical_stats': self.categorical_stats,
            'duplicates_count': self.duplicates_count,
            'validation_errors': self.validation_errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat()
        }
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the data quality report.
        
        Returns:
            String summary of the report
        """
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DATASET OVERVIEW:",
            f"  Total Records: {self.total_records:,}",
            f"  Total Features: {self.total_features}",
            f"  Duplicate Records: {self.duplicates_count:,}",
            ""
        ]
        
        # Missing values summary
        if self.missing_values:
            lines.append("MISSING VALUES:")
            cols_with_missing = {k: v for k, v in self.missing_values.items() if v > 0}
            if cols_with_missing:
                for col, count in sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True):
                    pct = self.missing_percentage.get(col, 0.0)
                    lines.append(f"  {col}: {count:,} ({pct:.2f}%)")
            else:
                lines.append("  No missing values found")
            lines.append("")
        
        # Data types summary
        if self.data_types:
            lines.append("DATA TYPES:")
            dtype_counts = {}
            for dtype in self.data_types.values():
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            for dtype, count in sorted(dtype_counts.items()):
                lines.append(f"  {dtype}: {count} columns")
            lines.append("")
        
        # Numerical statistics summary
        if self.numerical_stats:
            lines.append(f"NUMERICAL FEATURES: {len(self.numerical_stats)} columns")
            for col, stats in sorted(self.numerical_stats.items()):
                lines.append(f"  {col}:")
                lines.append(f"    Mean: {stats.get('mean', 0):.2f}, Std: {stats.get('std', 0):.2f}")
                lines.append(f"    Min: {stats.get('min', 0):.2f}, Max: {stats.get('max', 0):.2f}")
            lines.append("")
        
        # Categorical statistics summary
        if self.categorical_stats:
            lines.append(f"CATEGORICAL FEATURES: {len(self.categorical_stats)} columns")
            for col, value_counts in sorted(self.categorical_stats.items()):
                unique_count = len(value_counts)
                lines.append(f"  {col}: {unique_count} unique values")
                # Show top 3 values
                top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                for val, count in top_values:
                    lines.append(f"    '{val}': {count:,}")
            lines.append("")
        
        # Validation errors
        if self.validation_errors:
            lines.append("VALIDATION ERRORS:")
            for error in self.validation_errors:
                lines.append(f"  - {error}")
            lines.append("")
        
        # Warnings
        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DataQualityChecker:
    """Generates data quality reports for customer churn datasets."""
    
    def __init__(self):
        """Initialize the DataQualityChecker."""
        logger.info("DataQualityChecker initialized")
    
    def generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Generate a comprehensive data quality report for the given DataFrame.
        
        This method computes various statistics including:
        - Missing value counts and percentages
        - Data types for all columns
        - Numerical statistics (mean, std, min, max, quartiles)
        - Categorical value distributions
        - Duplicate record detection
        
        Args:
            df: pandas DataFrame to analyze
            
        Returns:
            DataQualityReport object containing all quality metrics
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")
        
        if df.empty:
            raise ValueError("DataFrame is empty (no rows)")
        
        logger.info(f"Generating quality report for DataFrame with {len(df)} records and {len(df.columns)} columns")
        
        # Basic metrics
        total_records = len(df)
        total_features = len(df.columns)
        
        # Missing values analysis
        missing_values = {}
        missing_percentage = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_values[col] = int(missing_count)
            missing_percentage[col] = float((missing_count / total_records) * 100) if total_records > 0 else 0.0
        
        # Data types
        data_types = {col: str(df[col].dtype) for col in df.columns}
        
        # Numerical statistics
        numerical_stats = {}
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            try:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    numerical_stats[col] = {
                        'count': int(len(col_data)),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        '25%': float(col_data.quantile(0.25)),
                        '50%': float(col_data.quantile(0.50)),
                        '75%': float(col_data.quantile(0.75))
                    }
                else:
                    numerical_stats[col] = {
                        'count': 0,
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        '25%': 0.0,
                        '50%': 0.0,
                        '75%': 0.0
                    }
            except Exception as e:
                logger.warning(f"Failed to compute statistics for numerical column '{col}': {e}")
                numerical_stats[col] = {}
        
        # Categorical statistics
        categorical_stats = {}
        categorical_columns = df.select_dtypes(include=['object', 'category', 'string']).columns
        for col in categorical_columns:
            try:
                value_counts = df[col].value_counts().to_dict()
                # Convert to regular Python types for JSON serialization
                categorical_stats[col] = {str(k): int(v) for k, v in value_counts.items()}
            except Exception as e:
                logger.warning(f"Failed to compute value counts for categorical column '{col}': {e}")
                categorical_stats[col] = {}
        
        # Duplicate detection
        duplicates_count = int(df.duplicated().sum())
        
        # Validation errors and warnings
        validation_errors = []
        warnings = []
        
        # Check for columns with high missing value percentage
        for col, pct in missing_percentage.items():
            if pct > 50:
                warnings.append(f"Column '{col}' has {pct:.2f}% missing values")
        
        # Check for duplicate records
        if duplicates_count > 0:
            warnings.append(f"Found {duplicates_count} duplicate records")
        
        # Check for columns with single unique value (potential constant columns)
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count == 1:
                warnings.append(f"Column '{col}' has only one unique value (constant column)")
        
        logger.info(f"Quality report generated: {duplicates_count} duplicates, {len(validation_errors)} errors, {len(warnings)} warnings")
        
        return DataQualityReport(
            total_records=total_records,
            total_features=total_features,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            data_types=data_types,
            numerical_stats=numerical_stats,
            categorical_stats=categorical_stats,
            duplicates_count=duplicates_count,
            validation_errors=validation_errors,
            warnings=warnings
        )
    
    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect duplicate records in the DataFrame.
        
        Args:
            df: pandas DataFrame to check for duplicates
            subset: Optional list of column names to consider for identifying duplicates.
                   If None, all columns are used.
            
        Returns:
            DataFrame containing only the duplicate records
            
        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")
        
        if df.empty:
            raise ValueError("DataFrame is empty (no rows)")
        
        logger.info(f"Detecting duplicates in DataFrame with {len(df)} records")
        
        # Find duplicate rows
        if subset:
            duplicates_mask = df.duplicated(subset=subset, keep=False)
            logger.info(f"Checking for duplicates based on columns: {subset}")
        else:
            duplicates_mask = df.duplicated(keep=False)
            logger.info("Checking for duplicates based on all columns")
        
        duplicate_records = df[duplicates_mask]
        
        logger.info(f"Found {len(duplicate_records)} duplicate records")
        
        return duplicate_records
