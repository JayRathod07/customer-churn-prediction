"""Demo script to showcase the Customer Churn Prediction System."""

import pandas as pd
from src.data.data_loader import DataLoader, DataQualityChecker
from src.utils.config import ConfigManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run demo of the customer churn prediction system."""
    
    print("=" * 80)
    print("CUSTOMER CHURN PREDICTION SYSTEM - DEMO")
    print("=" * 80)
    print()
    
    # Step 1: Load configuration
    print("Step 1: Loading configuration...")
    try:
        config_manager = ConfigManager('config/config.yaml')
        config = config_manager.config
        print(f"✓ Configuration loaded successfully")
        print(f"  - Data path: {config['data']['raw_data_path']}")
        print(f"  - Test size: {config['data']['test_size']}")
        print(f"  - Random state: {config['data']['random_state']}")
        print()
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return
    
    # Step 2: Load data
    print("Step 2: Loading customer churn data...")
    try:
        data_loader = DataLoader()
        df = data_loader.load_data(config['data']['raw_data_path'])
        print(f"✓ Data loaded successfully")
        print(f"  - Total records: {len(df):,}")
        print(f"  - Total features: {len(df.columns)}")
        print()
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return
    
    # Step 3: Validate schema
    print("Step 3: Validating data schema...")
    try:
        validation_result = data_loader.validate_schema(df)
        if validation_result.is_valid:
            print(f"✓ Schema validation passed")
            if validation_result.warnings:
                print(f"  - Warnings: {len(validation_result.warnings)}")
                for warning in validation_result.warnings[:3]:
                    print(f"    • {warning}")
        else:
            print(f"✗ Schema validation failed")
            print(f"  - Errors: {len(validation_result.errors)}")
            for error in validation_result.errors[:5]:
                print(f"    • {error}")
        print()
    except Exception as e:
        print(f"✗ Failed to validate schema: {e}")
        return
    
    # Step 4: Generate data quality report
    print("Step 4: Generating data quality report...")
    try:
        quality_checker = DataQualityChecker()
        quality_report = quality_checker.generate_quality_report(df)
        print(f"✓ Data quality report generated")
        print(f"  - Total records: {quality_report.total_records:,}")
        print(f"  - Total features: {quality_report.total_features}")
        print(f"  - Duplicate records: {quality_report.duplicates_count:,}")
        print()
        
        # Show missing values summary
        missing_cols = {k: v for k, v in quality_report.missing_values.items() if v > 0}
        if missing_cols:
            print(f"  Missing values found in {len(missing_cols)} columns:")
            for col, count in list(missing_cols.items())[:5]:
                pct = quality_report.missing_percentage[col]
                print(f"    • {col}: {count:,} ({pct:.2f}%)")
        else:
            print(f"  ✓ No missing values found")
        print()
    except Exception as e:
        print(f"✗ Failed to generate quality report: {e}")
        return
    
    # Step 5: Show data preview
    print("Step 5: Data preview...")
    print(df.head(5).to_string())
    print()
    
    # Step 6: Show churn distribution
    print("Step 6: Churn distribution...")
    churn_counts = df['churn'].value_counts()
    churn_pct = df['churn'].value_counts(normalize=True) * 100
    print(f"  - No churn: {churn_counts.get('No', 0):,} ({churn_pct.get('No', 0):.2f}%)")
    print(f"  - Churned: {churn_counts.get('Yes', 0):,} ({churn_pct.get('Yes', 0):.2f}%)")
    print()
    
    # Step 7: Show numerical statistics
    print("Step 7: Numerical feature statistics...")
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges']
    print(df[numerical_cols].describe().to_string())
    print()
    
    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Implement feature engineering (Task 4)")
    print("  2. Train machine learning models (Task 7)")
    print("  3. Build prediction API (Task 11)")
    print("  4. Deploy with Docker (Task 20)")
    print()


if __name__ == '__main__':
    main()
