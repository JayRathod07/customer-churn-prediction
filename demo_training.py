"""
Demo script for model training functionality.

This script demonstrates:
1. Loading and preparing data
2. Feature transformation
3. Training multiple models
4. Model evaluation and selection
5. Model persistence

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import pandas as pd
from src.data.data_loader import DataLoader
from src.features.feature_transformer import FeatureTransformer
from src.models.model_trainer import ModelTrainer
from src.utils.config import ConfigManager


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run the model training demo."""
    print("\n" + "=" * 70)
    print("  CUSTOMER CHURN PREDICTION - MODEL TRAINING DEMO")
    print("=" * 70)
    
    try:
        # Step 1: Load configuration
        print_section("Step 1: Loading Configuration")
        config_manager = ConfigManager()
        config = config_manager.config
        print("[OK] Configuration loaded successfully")
        
        # Step 2: Load data
        print_section("Step 2: Loading Data")
        loader = DataLoader()
        df = loader.load_data(config['data']['raw_data_path'])
        print(f"[OK] Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Step 3: Prepare features and target
        print_section("Step 3: Preparing Features and Target")
        
        # Separate features and target
        X = df.drop(['churn', 'customer_id'], axis=1)
        y = df['churn']
        
        print(f"[OK] Features shape: {X.shape}")
        print(f"[OK] Target shape: {y.shape}")
        print(f"[OK] Class distribution:")
        print(f"     - Churn (1): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
        print(f"     - No Churn (0): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
        
        # Step 4: Feature transformation
        print_section("Step 4: Feature Transformation")
        transformer = FeatureTransformer()
        X_transformed = transformer.fit_transform(X)
        
        print(f"[OK] Original features: {X.shape[1]}")
        print(f"[OK] Transformed features: {X_transformed.shape[1]}")
        print(f"[OK] Feature engineering complete")
        
        # Step 5: Initialize model trainer
        print_section("Step 5: Initializing Model Trainer")
        trainer = ModelTrainer(config)
        print("[OK] ModelTrainer initialized")
        
        # Step 6: Train-test split
        print_section("Step 6: Train-Test Split")
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
            X_transformed, y, stratify=True
        )
        
        print(f"[OK] Training set: {X_train.shape[0]} samples")
        print(f"[OK] Test set: {X_test.shape[0]} samples")
        print(f"[OK] Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"[OK] Test class distribution: {y_test.value_counts().to_dict()}")
        
        # Step 7: Train models (without hyperparameter tuning for demo speed)
        print_section("Step 7: Training Models")
        print("[INFO] Training models without hyperparameter tuning for demo speed...")
        print("[INFO] For production, set tune_hyperparameters=True")
        
        results = trainer.train_all_models(
            X_train, y_train, tune_hyperparameters=False
        )
        
        print(f"\n[OK] Trained {len(results)} models:")
        for model_name in results.keys():
            print(f"     - {model_name}")
        
        # Step 8: Evaluate models
        print_section("Step 8: Evaluating Models")
        
        evaluation_results = {}
        for model_name, result in results.items():
            print(f"\n[INFO] Evaluating {model_name}...")
            metrics = trainer.evaluate_model(result['model'], X_test, y_test)
            evaluation_results[model_name] = metrics
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Step 9: Select best model
        print_section("Step 9: Selecting Best Model")
        
        best_name, best_model, best_metrics = trainer.select_best_model(
            results, X_test, y_test, metric='f1'
        )
        
        print(f"[OK] Best model: {best_name}")
        print(f"[OK] Best F1-Score: {best_metrics['f1']:.4f}")
        print(f"\n[INFO] Best model metrics:")
        for metric_name, value in best_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Step 10: Save best model
        print_section("Step 10: Saving Best Model")
        
        model_path = trainer.save_model(
            best_model,
            best_name,
            best_metrics,
            metadata={
                'author': 'Jay Rathod',
                'dataset_size': len(X_train),
                'n_features': X_transformed.shape[1],
                'test_size': len(X_test)
            }
        )
        
        print(f"[OK] Model saved to: {model_path}")
        print(f"[OK] Metadata saved alongside model")
        
        # Step 11: Model comparison summary
        print_section("Step 11: Model Comparison Summary")
        
        comparison_df = pd.DataFrame(evaluation_results).T
        comparison_df = comparison_df.round(4)
        
        print("\n" + comparison_df.to_string())
        
        # Highlight best model for each metric
        print("\n[INFO] Best model for each metric:")
        for metric in comparison_df.columns:
            best_model_for_metric = comparison_df[metric].idxmax()
            best_score = comparison_df[metric].max()
            print(f"  {metric}: {best_model_for_metric} ({best_score:.4f})")
        
        # Final summary
        print_section("Demo Complete!")
        print(f"\n[OK] Successfully trained and evaluated {len(results)} models")
        print(f"[OK] Best model ({best_name}) saved to: {model_path}")
        print(f"[OK] Best F1-Score: {best_metrics['f1']:.4f}")
        
        # Check if model meets target
        target_f1 = 0.80
        if best_metrics['f1'] >= target_f1:
            print(f"[OK] Model meets target F1-score of {target_f1:.2f}")
        else:
            print(f"[WARNING] Model F1-score ({best_metrics['f1']:.4f}) below target ({target_f1:.2f})")
            print("[INFO] Consider:")
            print("  - Enabling hyperparameter tuning")
            print("  - Adding more training data")
            print("  - Feature engineering improvements")
        
        print("\n" + "=" * 70)
        print("  Thank you for using Customer Churn Prediction System!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
