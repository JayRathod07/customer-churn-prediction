"""
Complete Training Pipeline Script

Train models end-to-end: data loading, feature engineering, training, evaluation, and saving.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import argparse
import pandas as pd
from src.data.data_loader import DataLoader
from src.features.feature_transformer import FeatureTransformer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run complete training pipeline."""
    parser = argparse.ArgumentParser(description='Train Customer Churn Prediction Models')
    parser.add_argument('--data', type=str, default=None, help='Path to training data CSV')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--models', type=str, default='all', help='Models to train: all, lr, rf, gb')
    parser.add_argument('--report', action='store_true', help='Generate evaluation report')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 70)
    
    try:
        # Load configuration
        print_section("Step 1: Loading Configuration")
        config_manager = ConfigManager()
        config = config_manager.config
        print("[OK] Configuration loaded")
        
        # Load data
        print_section("Step 2: Loading Data")
        loader = DataLoader()
        data_path = args.data or config['data']['raw_data_path']
        df = loader.load_data(data_path)
        print(f"[OK] Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Validate schema
        validation_result = loader.validate_schema(df)
        if not validation_result.is_valid:
            print("[ERROR] Schema validation failed:")
            for error in validation_result.errors:
                print(f"  - {error}")
            return 1
        print("[OK] Schema validation passed")
        
        # Prepare features and target
        print_section("Step 3: Preparing Features and Target")
        X = df.drop(['churn', 'customer_id'], axis=1)
        y = df['churn']
        print(f"[OK] Features: {X.shape}")
        print(f"[OK] Target: {y.shape}")
        print(f"[OK] Class distribution: {dict(y.value_counts())}")
        
        # Feature engineering
        print_section("Step 4: Feature Engineering")
        transformer = FeatureTransformer()
        X_transformed = transformer.fit_transform(X)
        print(f"[OK] Original features: {X.shape[1]}")
        print(f"[OK] Transformed features: {X_transformed.shape[1]}")
        
        # Save transformer
        transformer.save('artifacts/feature_transformer.joblib')
        print("[OK] Transformer saved")
        
        # Train-test split
        print_section("Step 5: Train-Test Split")
        trainer = ModelTrainer(config)
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
            X_transformed, y, stratify=True
        )
        print(f"[OK] Training set: {X_train.shape[0]} samples")
        print(f"[OK] Test set: {X_test.shape[0]} samples")
        
        # Train models
        print_section("Step 6: Training Models")
        print(f"[INFO] Hyperparameter tuning: {'Enabled' if args.tune else 'Disabled'}")
        
        if args.models == 'all':
            results = trainer.train_all_models(X_train, y_train, tune_hyperparameters=args.tune)
            print(f"[OK] Trained {len(results)} models")
        else:
            results = {}
            if 'lr' in args.models:
                results['logistic_regression'] = trainer.train_logistic_regression(
                    X_train, y_train, tune_hyperparameters=args.tune
                )
            if 'rf' in args.models:
                results['random_forest'] = trainer.train_random_forest(
                    X_train, y_train, tune_hyperparameters=args.tune
                )
            if 'gb' in args.models:
                results['gradient_boosting'] = trainer.train_gradient_boosting(
                    X_train, y_train, tune_hyperparameters=args.tune
                )
            print(f"[OK] Trained {len(results)} models")
        
        # Evaluate models
        print_section("Step 7: Evaluating Models")
        evaluator = ModelEvaluator(config)
        
        for model_name, result in results.items():
            print(f"\n[INFO] Evaluating {model_name}...")
            metrics = evaluator.evaluate(result['model'], X_test, y_test, model_name)
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Select best model
        print_section("Step 8: Selecting Best Model")
        best_name, best_model, best_metrics = trainer.select_best_model(
            results, X_test, y_test, metric='f1'
        )
        
        print(f"[OK] Best model: {best_name}")
        print(f"[OK] F1-Score: {best_metrics['f1']:.4f}")
        print(f"[OK] ROC-AUC: {best_metrics['roc_auc']:.4f}")
        
        # Save best model
        print_section("Step 9: Saving Best Model")
        model_path = trainer.save_model(
            best_model,
            best_name,
            best_metrics,
            metadata={
                'author': 'Jay Rathod',
                'dataset_size': len(X_train),
                'n_features': X_transformed.shape[1],
                'hyperparameter_tuning': args.tune
            }
        )
        print(f"[OK] Model saved: {model_path}")
        
        # Generate report
        if args.report:
            print_section("Step 10: Generating Evaluation Report")
            y_pred = best_model.predict(X_test) if not hasattr(best_model, 'best_estimator_') else best_model.best_estimator_.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if not hasattr(best_model, 'best_estimator_') else best_model.best_estimator_.predict_proba(X_test)[:, 1]
            
            feature_names = transformer.get_feature_names()
            report_path = evaluator.generate_report(
                best_model, best_name, best_metrics, X_test, y_test,
                y_pred, y_pred_proba, feature_names, format='markdown'
            )
            print(f"[OK] Report saved: {report_path}")
        
        # Final summary
        print_section("Training Complete!")
        print(f"\n[OK] Best model: {best_name}")
        print(f"[OK] F1-Score: {best_metrics['f1']:.4f}")
        print(f"[OK] Model saved: {model_path}")
        
        # Check if target met
        target_f1 = 0.80
        if best_metrics['f1'] >= target_f1:
            print(f"[OK] Model meets target F1-score of {target_f1:.2f}")
        else:
            print(f"[WARNING] Model F1-score below target ({target_f1:.2f})")
            print("[INFO] Consider enabling hyperparameter tuning with --tune flag")
        
        print("\n" + "=" * 70 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
