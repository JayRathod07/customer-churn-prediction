"""Demo script for feature engineering functionality.

This script demonstrates the FeatureTransformer class including:
- Binary and categorical encoding
- Numerical scaling
- Derived feature creation
- Train-test splitting
- Transformer persistence

Author: Jay Rathod
"""

import pandas as pd
from src.data.data_loader import DataLoader
from src.features import FeatureTransformer, prepare_train_test_split
from src.utils.config import ConfigManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run feature engineering demo."""
    
    print("=" * 80)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 80)
    print()
    
    # Step 1: Load configuration and data
    print("Step 1: Loading data...")
    try:
        config = ConfigManager('config/config.yaml').config
        loader = DataLoader()
        df = loader.load_data(config['data']['raw_data_path'])
        print(f"[OK] Loaded {len(df)} records with {len(df.columns)} features")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    
    # Step 2: Create and fit transformer
    print("Step 2: Creating and fitting FeatureTransformer...")
    try:
        transformer = FeatureTransformer()
        print(f"  - Binary features: {len(transformer.binary_features)}")
        print(f"  - Categorical features: {len(transformer.categorical_features)}")
        print(f"  - Numerical features: {len(transformer.numerical_features)}")
        print()
        
        # Fit and transform
        df_transformed = transformer.fit_transform(df)
        print(f"[OK] Transformation complete")
        print(f"  - Original shape: {df.shape}")
        print(f"  - Transformed shape: {df_transformed.shape}")
        print(f"  - Total features after transformation: {len(transformer.get_feature_names())}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to transform data: {e}")
        return
    
    # Step 3: Show derived features
    print("Step 3: Derived features created...")
    derived_features = [
        'charges_per_month', 'service_count', 'tenure_group',
        'senior_with_dependents', 'high_value_customer'
    ]
    
    for feature in derived_features:
        if feature in df_transformed.columns:
            print(f"  - {feature}:")
            print(f"    Min: {df_transformed[feature].min():.2f}")
            print(f"    Max: {df_transformed[feature].max():.2f}")
            print(f"    Mean: {df_transformed[feature].mean():.2f}")
    print()
    
    # Step 4: Show feature statistics
    print("Step 4: Feature statistics...")
    print(f"  - Total features: {len(df_transformed.columns)}")
    print(f"  - Numerical features: {len([c for c in df_transformed.columns if df_transformed[c].dtype in ['int64', 'float64']])}")
    print(f"  - Binary features: {len([c for c in df_transformed.columns if df_transformed[c].nunique() == 2])}")
    print()
    
    # Step 5: Train-test split
    print("Step 5: Creating train-test split...")
    try:
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df_transformed,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'],
            stratify=True
        )
        
        print(f"[OK] Split complete")
        print(f"  - Training set: {len(X_train)} samples")
        print(f"  - Test set: {len(X_test)} samples")
        print(f"  - Train churn rate: {y_train.mean():.2%}")
        print(f"  - Test churn rate: {y_test.mean():.2%}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to split data: {e}")
        return
    
    # Step 6: Save transformer
    print("Step 6: Saving transformer...")
    try:
        transformer_path = 'artifacts/feature_transformer.joblib'
        transformer.save(transformer_path)
        print(f"[OK] Transformer saved to {transformer_path}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to save transformer: {e}")
        return
    
    # Step 7: Load and verify transformer
    print("Step 7: Loading and verifying transformer...")
    try:
        loaded_transformer = FeatureTransformer.load(transformer_path)
        print(f"[OK] Transformer loaded successfully")
        print(f"  - Is fitted: {loaded_transformer.is_fitted}")
        print(f"  - Feature count: {len(loaded_transformer.get_feature_names())}")
        
        # Verify it works
        df_test = loaded_transformer.transform(df.head(10))
        print(f"  - Test transformation: {df_test.shape}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load transformer: {e}")
        return
    
    # Step 8: Show sample transformed data
    print("Step 8: Sample transformed data (first 3 rows)...")
    print(df_transformed.head(3).to_string())
    print()
    
    print("=" * 80)
    print("FEATURE ENGINEERING DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Original features: {len(df.columns)}")
    print(f"  - Transformed features: {len(df_transformed.columns)}")
    print(f"  - Derived features: {len(derived_features)}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print()
    print("Next steps:")
    print("  1. Train machine learning models")
    print("  2. Evaluate model performance")
    print("  3. Build prediction API")
    print()


if __name__ == '__main__':
    main()
