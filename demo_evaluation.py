"""
Demo script for model evaluation functionality.

This script demonstrates:
1. Training a model
2. Comprehensive evaluation with metrics
3. Generating visualizations (confusion matrix, ROC curve, PR curve, feature importance)
4. Creating evaluation reports

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import pandas as pd
from src.data.data_loader import DataLoader
from src.features.feature_transformer import FeatureTransformer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.config import ConfigManager


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run the model evaluation demo."""
    print("\n" + "=" * 70)
    print("  CUSTOMER CHURN PREDICTION - MODEL EVALUATION DEMO")
    print("=" * 70)
    
    try:
        # Step 1: Load configuration and data
        print_section("Step 1: Loading Data and Configuration")
        config_manager = ConfigManager()
        config = config_manager.config
        
        loader = DataLoader()
        df = loader.load_data(config['data']['raw_data_path'])
        print(f"[OK] Loaded {len(df)} records")
        
        # Step 2: Prepare and transform features
        print_section("Step 2: Feature Engineering")
        X = df.drop(['churn', 'customer_id'], axis=1)
        y = df['churn']
        
        transformer = FeatureTransformer()
        X_transformed = transformer.fit_transform(X)
        print(f"[OK] Transformed features: {X_transformed.shape[1]}")
        
        # Step 3: Train model
        print_section("Step 3: Training Model")
        trainer = ModelTrainer(config)
        X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
            X_transformed, y, stratify=True
        )
        
        result = trainer.train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        model = result['model']
        print(f"[OK] Model trained: Logistic Regression")
        
        # Step 4: Initialize evaluator
        print_section("Step 4: Initializing Model Evaluator")
        evaluator = ModelEvaluator(config)
        print("[OK] ModelEvaluator initialized")
        
        # Step 5: Evaluate model
        print_section("Step 5: Evaluating Model")
        metrics = evaluator.evaluate(model, X_test, y_test, model_name='logistic_regression')
        
        print("\n[INFO] Model Performance Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper()}: {value:.4f}")
        
        # Step 6: Generate predictions
        print_section("Step 6: Generating Predictions")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        print(f"[OK] Generated predictions for {len(y_test)} samples")
        
        # Step 7: Create visualizations
        print_section("Step 7: Creating Visualizations")
        
        # Confusion Matrix
        cm_path = evaluator.plot_confusion_matrix(y_test, y_pred)
        print(f"[OK] Confusion matrix saved: {cm_path}")
        
        # ROC Curve
        roc_path = evaluator.plot_roc_curve(y_test, y_pred_proba)
        print(f"[OK] ROC curve saved: {roc_path}")
        
        # Precision-Recall Curve
        pr_path = evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
        print(f"[OK] PR curve saved: {pr_path}")
        
        # Feature Importance
        feature_names = transformer.get_feature_names()
        fi_path = evaluator.plot_feature_importance(model, feature_names, top_n=15)
        print(f"[OK] Feature importance saved: {fi_path}")
        
        # Step 8: Generate classification report
        print_section("Step 8: Classification Report")
        class_report = evaluator.generate_classification_report(y_test, y_pred)
        print(class_report)
        
        # Step 9: Generate comprehensive report
        print_section("Step 9: Generating Comprehensive Report")
        
        # Markdown report
        md_report_path = evaluator.generate_report(
            model, 'logistic_regression', metrics, X_test, y_test,
            y_pred, y_pred_proba, feature_names, format='markdown'
        )
        print(f"[OK] Markdown report saved: {md_report_path}")
        
        # JSON report
        json_report_path = evaluator.generate_report(
            model, 'logistic_regression', metrics, X_test, y_test,
            y_pred, y_pred_proba, feature_names, format='json'
        )
        print(f"[OK] JSON report saved: {json_report_path}")
        
        # Step 10: Feature importance analysis
        print_section("Step 10: Top 10 Most Important Features")
        importance_df = evaluator.extract_feature_importance(model, feature_names)
        
        if not importance_df.empty:
            top_10 = importance_df.head(10)
            print("\n" + top_10.to_string(index=False))
        
        # Final summary
        print_section("Demo Complete!")
        print(f"\n[OK] Model evaluated successfully")
        print(f"[OK] Best metric - ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"[OK] F1-Score: {metrics['f1']:.4f}")
        print(f"\n[INFO] Generated files:")
        print(f"  - Confusion Matrix: {cm_path}")
        print(f"  - ROC Curve: {roc_path}")
        print(f"  - PR Curve: {pr_path}")
        print(f"  - Feature Importance: {fi_path}")
        print(f"  - Markdown Report: {md_report_path}")
        print(f"  - JSON Report: {json_report_path}")
        
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
