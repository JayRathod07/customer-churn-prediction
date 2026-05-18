# Quick Start Guide - Customer Churn Prediction System

## 🎯 What You Have Now

A **working ML project foundation** with:
- ✅ Data generation (1,000 synthetic customer records)
- ✅ Data loading and validation
- ✅ Data quality reporting
- ✅ Configuration management
- ✅ Unit tests (9/9 passing)
- ✅ Complete project structure

---

## 🚀 Test Run Commands

### 1. Generate Data (1,000 records)
```bash
cd C:\Jay_PC\Desktop\project\customer-churn-prediction
.\venv\Scripts\python.exe scripts\generate_data.py --n-samples 1000
```

**Expected Output:**
```
Generating 1000 synthetic customer records...
✓ Data saved to data\customer_churn.csv

Dataset Statistics:
  Total records: 1000
  Churn rate: 47.50%
  Features: 21
```

---

### 2. Run Demo Script
```bash
.\venv\Scripts\python.exe demo.py
```

**What It Shows:**
- ✅ Configuration loading
- ✅ Data loading (1,000 records)
- ✅ Schema validation
- ✅ Data quality report
- ✅ Data preview
- ✅ Churn distribution
- ✅ Numerical statistics

---

### 3. Run Unit Tests
```bash
.\venv\Scripts\python.exe -m pytest tests/test_data_loader.py -v
```

**Expected Result:**
```
9 passed in 2.11s
```

**Tests Covered:**
- ✅ Load data from CSV
- ✅ Handle missing files
- ✅ Handle empty files
- ✅ Validate schema
- ✅ Detect missing columns
- ✅ Detect wrong data types
- ✅ Handle extra columns
- ✅ Validate empty DataFrames
- ✅ Data type compatibility

---

### 4. Quick Data Check
```bash
.\venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/customer_churn.csv'); print(f'Records: {len(df)}'); print(f'Features: {len(df.columns)}'); print(f'Churn rate: {(df[\"churn\"] == \"Yes\").sum() / len(df) * 100:.2f}%')"
```

---

### 5. Test Data Loader Directly
```bash
.\venv\Scripts\python.exe -c "from src.data.data_loader import DataLoader; loader = DataLoader(); df = loader.load_data('data/customer_churn.csv'); result = loader.validate_schema(df); print(f'Valid: {result.is_valid}'); print(f'Records: {len(df)}')"
```

---

### 6. Test Configuration Manager
```bash
.\venv\Scripts\python.exe -c "from src.utils.config import ConfigManager; config = ConfigManager('config/config.yaml'); print(f'Data path: {config.config[\"data\"][\"raw_data_path\"]}'); print(f'Test size: {config.config[\"data\"][\"test_size\"]}')"
```

---

### 7. Generate Quality Report
```bash
.\venv\Scripts\python.exe -c "from src.data.data_loader import DataLoader, DataQualityChecker; import pandas as pd; df = pd.read_csv('data/customer_churn.csv'); checker = DataQualityChecker(); report = checker.generate_quality_report(df); print(report.summary())"
```

---

## 📊 What Each Component Does

### 1. **Data Generation** (`scripts/generate_data.py`)
- Creates realistic synthetic customer data
- 21 features: demographics, services, charges
- Realistic churn patterns
- Configurable sample size

**Features Generated:**
- Customer ID
- Demographics (gender, senior_citizen, partner, dependents)
- Tenure (months with company)
- Services (phone, internet, streaming, etc.)
- Charges (monthly, total)
- Churn label (Yes/No)

---

### 2. **Data Loader** (`src/data/data_loader.py`)
- Loads CSV files into pandas DataFrames
- Validates schema (columns and data types)
- Handles errors gracefully
- Provides detailed error messages

**Key Methods:**
- `load_data(file_path)` - Load CSV file
- `validate_schema(df)` - Validate DataFrame schema
- `handle_missing_values(df, strategy)` - Handle missing data

---

### 3. **Data Quality Checker** (`src/data/data_loader.py`)
- Generates comprehensive quality reports
- Detects missing values
- Identifies duplicates
- Computes statistics

**Metrics:**
- Missing value counts and percentages
- Data type distribution
- Numerical statistics (mean, std, min, max, quartiles)
- Categorical value distributions
- Duplicate record count

---

### 4. **Configuration Manager** (`src/utils/config.py`)
- Loads YAML configuration files
- Validates required fields
- Supports environment variables
- Provides easy access to settings

**Configuration Sections:**
- Data paths and parameters
- Feature engineering options
- Model hyperparameters
- Training parameters
- API settings
- Storage paths
- Logging configuration

---

## 🔍 Verify Everything Works

Run this complete test sequence:

```bash
# 1. Generate fresh data
.\venv\Scripts\python.exe scripts\generate_data.py --n-samples 1000

# 2. Run demo
.\venv\Scripts\python.exe demo.py

# 3. Run tests
.\venv\Scripts\python.exe -m pytest tests/test_data_loader.py -v

# 4. Check data file exists
dir data\customer_churn.csv
```

**Expected Results:**
- ✅ Data file created (1,000 records)
- ✅ Demo runs successfully
- ✅ All 9 tests pass
- ✅ No errors or warnings

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/                           # Data storage
│   └── customer_churn.csv          # Generated data (1,000 records)
│
├── config/                         # Configuration
│   └── config.yaml                 # System configuration
│
├── src/                            # Source code
│   ├── data/                       # Data modules
│   │   ├── data_loader.py          # ✅ DataLoader & DataQualityChecker
│   │   └── __init__.py
│   │
│   ├── utils/                      # Utilities
│   │   ├── config.py               # ✅ Configuration manager
│   │   ├── logger.py               # ✅ Logging setup
│   │   ├── exceptions.py           # ✅ Custom exceptions
│   │   └── __init__.py
│   │
│   ├── features/                   # Feature engineering (TODO)
│   ├── models/                     # ML models (TODO)
│   └── api/                        # FastAPI (TODO)
│
├── tests/                          # Unit tests
│   └── test_data_loader.py         # ✅ 9 tests passing
│
├── scripts/                        # Utility scripts
│   └── generate_data.py            # ✅ Data generation
│
├── requirements.txt                # ✅ Dependencies installed
├── demo.py                         # ✅ Demo script
├── PROJECT_STATUS.md               # ✅ Detailed status
├── QUICK_START.md                  # ✅ This file
└── venv/                           # Virtual environment
```

---

## 🎓 Understanding the Code

### Example: Load and Validate Data

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
    print("✗ Schema validation failed:")
    for error in result.errors:
        print(f"  - {error}")
```

### Example: Generate Quality Report

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

# Access specific metrics
print(f"Total records: {report.total_records}")
print(f"Duplicates: {report.duplicates_count}")
```

### Example: Load Configuration

```python
from src.utils.config import ConfigManager

# Load config
config = ConfigManager('config/config.yaml')

# Access settings
data_path = config.get('data.raw_data_path')
test_size = config.get('data.test_size')
random_state = config.get('data.random_state')

print(f"Data path: {data_path}")
print(f"Test size: {test_size}")
print(f"Random state: {random_state}")
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:**
```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Issue: "FileNotFoundError: data/customer_churn.csv"
**Solution:**
```bash
.\venv\Scripts\python.exe scripts\generate_data.py --n-samples 1000
```

### Issue: Tests fail with import errors
**Solution:**
```bash
# Make sure you're in the project root
cd C:\Jay_PC\Desktop\project\customer-churn-prediction

# Run tests from project root
.\venv\Scripts\python.exe -m pytest tests/test_data_loader.py -v
```

### Issue: "No such file or directory: 'config/config.yaml'"
**Solution:**
```bash
# Make sure you're in the project root
cd C:\Jay_PC\Desktop\project\customer-churn-prediction

# Check if config exists
dir config\config.yaml
```

---

## 📈 Next Steps

1. **Review the code** - Understand how each module works
2. **Run all test commands** - Verify everything works
3. **Read PROJECT_STATUS.md** - See what's next
4. **Check tasks.md** - See the complete implementation plan
5. **Start implementing** - Follow the task list

---

## 🎯 Key Achievements

✅ **Working Data Pipeline**
- Generate → Load → Validate → Report

✅ **Production-Ready Code**
- Error handling
- Logging
- Configuration management
- Unit tests

✅ **Clean Architecture**
- Modular design
- Separation of concerns
- Reusable components

✅ **Documentation**
- Docstrings
- Type hints
- Test coverage

---

## 📞 Quick Reference

| Command | Purpose |
|---------|---------|
| `python demo.py` | Run full demo |
| `python scripts\generate_data.py` | Generate data |
| `pytest tests/ -v` | Run all tests |
| `python -m pytest tests/test_data_loader.py::TestDataLoader -v` | Run specific test class |

---

**Ready to continue?** Check `PROJECT_STATUS.md` for the next phase!
