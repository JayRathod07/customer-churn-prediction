"""Comprehensive test script to verify all implemented features."""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        import pandas as pd
        print("✓ pandas imported")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        from src.data.data_loader import DataLoader, DataQualityChecker, ValidationResult
        print("✓ DataLoader imported")
        print("✓ DataQualityChecker imported")
        print("✓ ValidationResult imported")
    except ImportError as e:
        print(f"✗ Failed to import data modules: {e}")
        return False
    
    try:
        from src.utils.config import ConfigManager
        print("✓ ConfigManager imported")
    except ImportError as e:
        print(f"✗ Failed to import ConfigManager: {e}")
        return False
    
    try:
        from src.utils.exceptions import DataLoadError, DataValidationError
        print("✓ Custom exceptions imported")
    except ImportError as e:
        print(f"✗ Failed to import exceptions: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_data_generation():
    """Test data generation."""
    print("=" * 80)
    print("TEST 2: Data Generation")
    print("=" * 80)
    
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/generate_data.py", "--n-samples", "100", "--output", "data/test_data.csv"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✓ Data generation script executed successfully")
            print(f"  Output: {result.stdout.strip()[:200]}")
            
            # Check if file was created
            if Path("data/test_data.csv").exists():
                print("✓ Test data file created")
                return True
            else:
                print("✗ Test data file not found")
                return False
        else:
            print(f"✗ Data generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error during data generation: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    print("\n" + "=" * 80)
    print("TEST 3: Data Loading")
    print("=" * 80)
    
    from src.data.data_loader import DataLoader
    import pandas as pd
    
    try:
        loader = DataLoader()
        print("✓ DataLoader instance created")
        
        # Test loading existing data
        df = loader.load_data("data/test_data.csv")
        print(f"✓ Data loaded successfully: {len(df)} records, {len(df.columns)} columns")
        
        # Test loading non-existent file
        try:
            loader.load_data("data/nonexistent.csv")
            print("✗ Should have raised DataLoadError for missing file")
            return False
        except Exception as e:
            print(f"✓ Correctly raised error for missing file: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False


def test_schema_validation():
    """Test schema validation."""
    print("\n" + "=" * 80)
    print("TEST 4: Schema Validation")
    print("=" * 80)
    
    from src.data.data_loader import DataLoader
    import pandas as pd
    
    try:
        loader = DataLoader()
        df = loader.load_data("data/test_data.csv")
        
        # Test valid schema
        result = loader.validate_schema(df)
        if result.is_valid:
            print(f"✓ Schema validation passed")
        else:
            print(f"✗ Schema validation failed: {result.errors}")
            return False
        
        # Test invalid schema (missing columns)
        df_invalid = df.drop(columns=['tenure', 'monthly_charges'])
        result = loader.validate_schema(df_invalid)
        if not result.is_valid:
            print(f"✓ Correctly detected missing columns: {len(result.errors)} errors")
        else:
            print(f"✗ Should have detected missing columns")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Schema validation test failed: {e}")
        return False


def test_quality_report():
    """Test data quality reporting."""
    print("\n" + "=" * 80)
    print("TEST 5: Data Quality Reporting")
    print("=" * 80)
    
    from src.data.data_loader import DataQualityChecker
    import pandas as pd
    
    try:
        df = pd.read_csv("data/test_data.csv")
        checker = DataQualityChecker()
        
        report = checker.generate_quality_report(df)
        print(f"✓ Quality report generated")
        print(f"  - Total records: {report.total_records}")
        print(f"  - Total features: {report.total_features}")
        print(f"  - Duplicates: {report.duplicates_count}")
        print(f"  - Errors: {len(report.validation_errors)}")
        print(f"  - Warnings: {len(report.warnings)}")
        
        # Check report has required attributes
        assert hasattr(report, 'total_records')
        assert hasattr(report, 'total_features')
        assert hasattr(report, 'missing_values')
        assert hasattr(report, 'numerical_stats')
        assert hasattr(report, 'categorical_stats')
        print("✓ Report has all required attributes")
        
        return True
    except Exception as e:
        print(f"✗ Quality report test failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\n" + "=" * 80)
    print("TEST 6: Configuration Management")
    print("=" * 80)
    
    from src.utils.config import ConfigManager
    
    try:
        config = ConfigManager("config/config.yaml")
        print("✓ Configuration loaded")
        
        # Test accessing config values
        data_path = config.get('data.raw_data_path')
        print(f"✓ Data path: {data_path}")
        
        test_size = config.get('data.test_size')
        print(f"✓ Test size: {test_size}")
        
        random_state = config.get('data.random_state')
        print(f"✓ Random state: {random_state}")
        
        # Test nested access
        models = config.get('training.models')
        print(f"✓ Models config: {len(models)} models configured")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_missing_value_handling():
    """Test missing value handling."""
    print("\n" + "=" * 80)
    print("TEST 7: Missing Value Handling")
    print("=" * 80)
    
    from src.data.data_loader import DataLoader
    import pandas as pd
    import numpy as np
    
    try:
        loader = DataLoader()
        
        # Create test data with missing values
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1.0, np.nan, 3.0, 4.0, 5.0],
            'col3': ['A', 'B', np.nan, 'A', 'C']
        })
        
        print(f"✓ Test data created with {df.isnull().sum().sum()} missing values")
        
        # Test mean imputation
        df_mean = loader._impute_mean(df.copy(), ['col1', 'col2'])
        if df_mean['col1'].isnull().sum() == 0:
            print("✓ Mean imputation works")
        else:
            print("✗ Mean imputation failed")
            return False
        
        # Test median imputation
        df_median = loader._impute_median(df.copy(), ['col1', 'col2'])
        if df_median['col1'].isnull().sum() == 0:
            print("✓ Median imputation works")
        else:
            print("✗ Median imputation failed")
            return False
        
        # Test mode imputation
        df_mode = loader._impute_mode(df.copy(), ['col3'])
        if df_mode['col3'].isnull().sum() == 0:
            print("✓ Mode imputation works")
        else:
            print("✗ Mode imputation failed")
            return False
        
        # Test drop missing
        df_drop = loader._drop_missing(df.copy(), ['col1', 'col2', 'col3'])
        if len(df_drop) < len(df):
            print(f"✓ Drop missing works: {len(df)} → {len(df_drop)} rows")
        else:
            print("✗ Drop missing failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Missing value handling test failed: {e}")
        return False


def test_unit_tests():
    """Run pytest unit tests."""
    print("\n" + "=" * 80)
    print("TEST 8: Unit Tests (pytest)")
    print("=" * 80)
    
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_data_loader.py::TestDataLoader", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if "passed" in result.stdout:
            # Extract pass count
            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line:
                    print(f"✓ {line.strip()}")
                    break
            return True
        else:
            print(f"✗ Some tests failed")
            print(result.stdout[-500:])  # Last 500 chars
            return False
    except Exception as e:
        print(f"✗ Unit tests failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "CUSTOMER CHURN PREDICTION SYSTEM" + " " * 31 + "║")
    print("║" + " " * 25 + "COMPREHENSIVE TEST SUITE" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Data Loading", test_data_loading),
        ("Schema Validation", test_schema_validation),
        ("Quality Reporting", test_quality_report),
        ("Configuration", test_configuration),
        ("Missing Value Handling", test_missing_value_handling),
        ("Unit Tests", test_unit_tests),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {test_name}")
    
    print("=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your project is working correctly! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
