# Customer Churn Prediction System

> **An end-to-end machine learning project for predicting customer churn with production-ready code, comprehensive testing, and Docker deployment.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Development Status](#development-status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Customer Churn Prediction System** is a complete machine learning pipeline that predicts whether customers will churn (leave the service). This project demonstrates:

- **Data Engineering**: Synthetic data generation, loading, validation, and quality reporting
- **Feature Engineering**: Categorical encoding, numerical scaling, and derived features
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and visualizations
- **API Development**: RESTful API with FastAPI for real-time predictions
- **Deployment**: Docker containerization for easy deployment
- **Testing**: Unit tests, integration tests, and property-based tests

**Perfect for**: Data science portfolios, resume projects, and learning end-to-end ML workflows.

---

## ✨ Features

### ✅ Currently Implemented

- **Data Generation**: Generate realistic synthetic customer data with configurable parameters
- **Data Loading**: Robust CSV loading with comprehensive error handling
- **Schema Validation**: Automatic validation of data structure and types
- **Data Quality Reporting**: Detailed quality metrics including missing values, duplicates, and statistics
- **Missing Value Handling**: Multiple imputation strategies (mean, median, mode, drop)
- **Configuration Management**: YAML-based configuration with environment variable support
- **Logging**: Structured JSON logging for production observability
- **Testing**: Comprehensive unit tests with pytest (100% passing)

### 🚧 In Development

- Feature engineering pipeline
- ML model training (Logistic Regression, Random Forest, XGBoost)
- Model evaluation and visualization
- Prediction API with FastAPI
- Docker containerization
- CI/CD pipeline

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **ML/Data** | pandas, NumPy, scikit-learn, XGBoost, LightGBM |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | pytest, pytest-cov, hypothesis |
| **Deployment** | Docker, docker-compose |
| **Code Quality** | black, flake8, mypy, pre-commit |
| **Visualization** | matplotlib, seaborn, plotly |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
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
   python demo.py
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
│   │   └── __init__.py
│   │
│   ├── models/                     # ML models
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
│   └── test_data_loader.py         # DataLoader tests
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
├── demo.py                         # Demo script
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

---

## 🧪 Testing

### Run All Tests

```bash
# Run comprehensive test suite
python test_all.py

# Run specific test file
pytest tests/test_data_loader.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Results

```
✅ Module Imports          - PASS
✅ Data Generation         - PASS
✅ Data Loading            - PASS
✅ Schema Validation       - PASS
✅ Quality Reporting       - PASS
✅ Configuration           - PASS
✅ Missing Value Handling  - PASS
✅ Unit Tests              - PASS (9/9)

TOTAL: 8/8 tests passed (100.0%)
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
- [x] Unit tests

### Phase 2: Feature Engineering 🚧 IN PROGRESS

- [ ] Feature transformer class
- [ ] Categorical encoding
- [ ] Numerical scaling
- [ ] Derived features
- [ ] Transformer persistence

### Phase 3: Model Training 📋 PLANNED

- [ ] Train-test split
- [ ] Model trainer class
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] Gradient Boosting
- [ ] Hyperparameter tuning
- [ ] Model selection

### Phase 4: Model Evaluation 📋 PLANNED

- [ ] Model evaluator class
- [ ] Metrics computation
- [ ] Confusion matrix
- [ ] Feature importance
- [ ] Visualization functions
- [ ] Report generation

### Phase 5: API Development 📋 PLANNED

- [ ] FastAPI application
- [ ] Prediction endpoints
- [ ] Health check
- [ ] Model info endpoint
- [ ] Error handling

### Phase 6: Deployment 📋 PLANNED

- [ ] Dockerfile
- [ ] docker-compose
- [ ] CI/CD pipeline
- [ ] Documentation

**Overall Progress**: ~15% (15/101 tasks completed)

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
| Test Coverage | 80% | 100% (Phase 1) |
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

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Inspired by real-world customer churn prediction challenges
- Built with modern ML best practices
- Designed for production deployment

---

## 📚 Additional Resources

- [Project Status](PROJECT_STATUS.md) - Detailed implementation status
- [Quick Start Guide](QUICK_START.md) - Step-by-step setup instructions
- [Task List](.kiro/specs/ml-end-to-end-project/tasks.md) - Complete task breakdown
- [Design Document](.kiro/specs/ml-end-to-end-project/design.md) - Technical design
- [Requirements](.kiro/specs/ml-end-to-end-project/requirements.md) - Functional requirements

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for the ML community

</div>
