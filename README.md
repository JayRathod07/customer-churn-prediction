# Customer Churn Prediction System

An end-to-end machine learning project for predicting customer churn with production-ready code, comprehensive testing, and Docker deployment.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

This project demonstrates a complete ML pipeline for predicting customer churn. It includes everything from data generation and validation to model training and API deployment. The focus is on production-ready code with proper error handling, logging, testing, and configuration management.

**What's included:**
- Data engineering pipeline with validation and quality reporting
- Feature engineering with multiple encoding strategies
- Model training with hyperparameter tuning
- RESTful API for predictions
- Docker containerization
- Comprehensive test suite

---

## Features

### Currently Working

- **Data Generation**: Creates realistic synthetic customer data with 21 features
- **Data Loading & Validation**: Robust CSV loading with schema validation
- **Data Quality Reporting**: Detailed metrics on missing values, duplicates, and statistics
- **Missing Value Handling**: Multiple imputation strategies (mean, median, mode, constant)
- **Feature Engineering**: Complete transformation pipeline with encoding, scaling, and derived features
- **Data Models**: Pydantic V2 models for robust data validation and API contracts
- **Model Training**: Train multiple ML models (Logistic Regression, Random Forest, Gradient Boosting) with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics, confusion matrix, ROC/PR curves, feature importance, and automated reports
- **Model Persistence**: Save/load trained models with metadata
- **Visualization**: Automated generation of evaluation plots and charts
- **Configuration Management**: YAML-based config with environment variable support
- **Logging**: Structured logging for production environments
- **Testing**: Comprehensive unit tests with pytest (120 tests passing)

### In Development

- Prediction API with FastAPI
- Docker containerization

---

## Tech Stack

- **Language**: Python 3.11+
- **ML/Data**: pandas, NumPy, scikit-learn, XGBoost, LightGBM
- **API**: FastAPI, Uvicorn, Pydantic
- **Testing**: pytest, pytest-cov, hypothesis
- **Deployment**: Docker, docker-compose
- **Code Quality**: black, flake8, mypy
- **Visualization**: matplotlib, seaborn, plotly

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jayRathod07/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate synthetic data**
   ```bash
   python scripts/generate_data.py --n-samples 10000
   ```

5. **Run demo**
   ```bash
   # Data pipeline demo
   python demo.py
   
   # Feature engineering demo
   python demo_features.py
   
   # Model training demo
   python demo_training.py
   ```

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/                           # Data storage
│   └── customer_churn.csv          # Generated customer data
│
├── config/                         # Configuration files
│   └── config.yaml                 # System configuration
│
├── src/                            # Source code
│   ├── data/                       # Data modules
│   │   ├── data_loader.py          # Data loading and validation
│   │   └── __init__.py
│   │
│   ├── features/                   # Feature engineering
│   │   ├── feature_transformer.py  # Feature transformation pipeline
│   │   └── __init__.py
│   │
│   ├── models/                     # ML models and data models
│   │   ├── data_models.py          # Pydantic validation models
│   │   ├── model_trainer.py        # Model training pipeline
│   │   └── __init__.py
│   │
│   ├── api/                        # FastAPI endpoints
│   │   └── __init__.py
│   │
│   └── utils/                      # Utilities
│       ├── config.py               # Configuration manager
│       ├── logger.py               # Logging setup
│       ├── exceptions.py           # Custom exceptions
│       └── __init__.py
│
├── tests/                          # Unit tests
│   ├── test_data_loader.py         # DataLoader tests (24 tests)
│   ├── test_feature_transformer.py # Feature tests (25 tests)
│   ├── test_data_models.py         # Pydantic model tests (26 tests)
│   └── test_model_trainer.py       # Model training tests (25 tests)
│
├── scripts/                        # Utility scripts
│   └── generate_data.py            # Data generation
│
├── notebooks/                      # Jupyter notebooks
│
├── models/                         # Saved models
├── artifacts/                      # Model artifacts
├── reports/                        # Evaluation reports
├── logs/                           # Application logs
│
├── requirements.txt                # Python dependencies
├── demo.py                         # Demo script (data pipeline)
├── demo_features.py                # Demo script (feature engineering)
├── demo_training.py                # Demo script (model training)
├── test_all.py                     # Comprehensive test suite
├── PROJECT_STATUS.md               # Detailed project status
├── QUICK_START.md                  # Quick start guide
└── README.md                       # This file
```

---

## 💻 Usage

### Generate Synthetic Data

```bash
# Generate 10,000 customer records
python scripts/generate_data.py --n-samples 10000

# Specify output path
python scripts/generate_data.py --n-samples 5000 --output data/my_data.csv

# Set random seed for reproducibility
python scripts/generate_data.py --n-samples 1000 --random-state 42
```

### Load and Validate Data

```python
from src.data.data_loader import DataLoader

# Create loader
loader = DataLoader()

# Load data
df = loader.load_data('data/customer_churn.csv')
print(f"Loaded {len(df)} records")

# Validate schema
result = loader.validate_schema(df)
if result.is_valid:
    print("✓ Schema is valid")
else:
    print("✗ Validation errors:")
    for error in result.errors:
        print(f"  - {error}")
```

### Generate Quality Report

```python
from src.data.data_loader import DataQualityChecker
import pandas as pd

# Load data
df = pd.read_csv('data/customer_churn.csv')

# Generate report
checker = DataQualityChecker()
report = checker.generate_quality_report(df)

# Print summary
print(report.summary())
```

### Handle Missing Values

```python
from src.data.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data('data/customer_churn.csv')

# Define strategy
strategy = {
    'numerical': 'median',  # mean, median, drop
    'categorical': 'mode'   # mode, constant, drop
}

# Handle missing values
df_clean = loader.handle_missing_values(df, strategy)
```

### Feature Engineering

```python
from src.features.feature_transformer import FeatureTransformer
import pandas as pd

# Load data
df = pd.read_csv('data/customer_churn.csv')

# Create and fit transformer
transformer = FeatureTransformer()
X_transformed = transformer.fit_transform(df)

print(f"Original features: {df.shape[1]}")
print(f"Transformed features: {X_transformed.shape[1]}")

# Save transformer for later use
transformer.save('artifacts/feature_transformer.joblib')

# Load transformer
loaded_transformer = FeatureTransformer.load('artifacts/feature_transformer.joblib')
X_new = loaded_transformer.transform(new_data)
```

### Data Validation with Pydantic

```python
from src.models.data_models import CustomerData, PredictionRequest

# Validate customer data
customer = CustomerData(
    customer_id="CUST001",
    gender="Male",
    senior_citizen=0,
    partner="Yes",
    dependents="No",
    tenure=12,
    phone_service="Yes",
    multiple_lines="No",
    internet_service="Fiber optic",
    online_security="No",
    online_backup="Yes",
    device_protection="No",
    tech_support="No",
    streaming_tv="Yes",
    streaming_movies="No",
    contract="Month-to-month",
    paperless_billing="Yes",
    payment_method="Electronic check",
    monthly_charges=70.35,
    total_charges=840.75
)

# Create prediction request
request = PredictionRequest(
    customer_id="CUST001",
    features=customer.model_dump(exclude={'customer_id'})
)
```

### Model Training

```python
from src.data.data_loader import DataLoader
from src.features.feature_transformer import FeatureTransformer
from src.models.model_trainer import ModelTrainer
from src.utils.config import ConfigManager

# Load configuration
config_manager = ConfigManager()
config = config_manager.config

# Load and prepare data
loader = DataLoader()
df = loader.load_data('data/customer_churn.csv')

# Separate features and target
X = df.drop(['churn', 'customer_id'], axis=1)
y = df['churn']

# Transform features
transformer = FeatureTransformer()
X_transformed = transformer.fit_transform(X)

# Initialize trainer
trainer = ModelTrainer(config)

# Train-test split
X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(
    X_transformed, y, stratify=True
)

# Train all models
results = trainer.train_all_models(X_train, y_train, tune_hyperparameters=True)

# Select best model
best_name, best_model, best_metrics = trainer.select_best_model(
    results, X_test, y_test, metric='f1'
)

print(f"Best model: {best_name}")
print(f"F1-Score: {best_metrics['f1']:.4f}")

# Save best model
model_path = trainer.save_model(best_model, best_name, best_metrics)
print(f"Model saved to: {model_path}")
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run comprehensive test suite
python test_all.py

# Run specific test file
pytest tests/test_data_loader.py -v
pytest tests/test_feature_transformer.py -v
pytest tests/test_data_models.py -v
pytest tests/test_model_trainer.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Results

```
✅ Data Loader Tests       - PASS (24/24)
✅ Feature Transformer     - PASS (25/25)
✅ Data Models Tests       - PASS (26/26)
✅ Model Trainer Tests     - PASS (25/25)
✅ Model Evaluator Tests   - PASS (20/20)

TOTAL: 120/120 tests passed (100.0%)
```

---

## 📊 Development Status

### Phase 1: Data Infrastructure ✅ COMPLETE

- [x] Project structure setup
- [x] Data generation script
- [x] Data loading and validation
- [x] Data quality reporting
- [x] Missing value handling
- [x] Configuration management
- [x] Logging setup
- [x] Unit tests (24 tests passing)

### Phase 2: Feature Engineering ✅ COMPLETE

- [x] Feature transformer class
- [x] Categorical encoding (binary and one-hot)
- [x] Numerical scaling (StandardScaler)
- [x] Derived features (5 new features)
- [x] Transformer persistence (save/load)
- [x] Unit tests (25 tests passing)
- [x] Demo script

### Phase 3: Data Models ✅ COMPLETE

- [x] Pydantic V2 models for validation
- [x] CustomerData and TrainingData models
- [x] API request/response models
- [x] ModelMetadata and EvaluationMetrics models
- [x] Enum classes for categorical fields
- [x] Unit tests (26 tests passing)

### Phase 4: Model Training ✅ COMPLETE

- [x] Train-test split with stratification
- [x] ModelTrainer class
- [x] Logistic Regression training
- [x] Random Forest training
- [x] Gradient Boosting training
- [x] Hyperparameter tuning (RandomizedSearchCV)
- [x] Model selection logic
- [x] Model persistence with metadata
- [x] Unit tests (25 tests passing)
- [x] Demo script

### Phase 5: Model Evaluation ✅ COMPLETE

- [x] ModelEvaluator class
- [x] Comprehensive metrics computation
- [x] Confusion matrix generation and plotting
- [x] ROC curve and PR curve visualization
- [x] Feature importance extraction and plotting
- [x] Classification report generation
- [x] Evaluation reports (Markdown, JSON)
- [x] Unit tests (20 tests passing)
- [x] Demo script

### Phase 6: API Development 📋 PLANNED

- [ ] FastAPI application
- [ ] Prediction endpoints
- [ ] Health check
- [ ] Model info endpoint
- [ ] Error handling

### Phase 7: Deployment 📋 PLANNED

- [ ] Dockerfile
- [ ] docker-compose
- [ ] CI/CD pipeline
- [ ] Documentation

**Overall Progress**: ~55% (60/101 tasks completed)

---

## 🗺️ Roadmap

### Short Term (Next 2 Weeks)

- Complete feature engineering module
- Implement model training pipeline
- Add model evaluation and visualization
- Create prediction API

### Medium Term (Next Month)

- Docker containerization
- CI/CD pipeline setup
- Comprehensive documentation
- Example notebooks

### Long Term (Next 3 Months)

- Model monitoring and drift detection
- A/B testing framework
- Advanced feature engineering
- Model explainability (SHAP, LIME)

---

## 📈 Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Test Coverage | 80% | 100% (Phases 1-3) |
| Model F1-Score | 80% | TBD |
| API Response Time | <100ms | TBD |
| Docker Image Size | <2GB | TBD |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Run tests before submitting PR

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Jay Rathod**

- GitHub: [@jayRathod07](https://github.com/jayRathod07)
- LinkedIn: [Jay Rathod](https://www.linkedin.com/in/jay-rathod-9ab3a0371/)
- Email: jayrathod121005@gmail.com

---

## 🙏 Acknowledgments

- Inspired by real-world customer churn prediction challenges
- Built with modern ML best practices
- Designed for production deployment

---

## 📚 Additional Resources

- [Project Status](PROJECT_STATUS.md) - Detailed implementation status
- [Quick Start Guide](QUICK_START.md) - Step-by-step setup instructions

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for the ML community

</div>
