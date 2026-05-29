"""
API Usage Examples

Demonstrates how to use the Customer Churn Prediction API.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import requests
import json


# API Base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_health_check():
    """Example: Health check."""
    print_section("Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def example_model_info():
    """Example: Get model information."""
    print_section("Model Information")
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def example_single_prediction():
    """Example: Single prediction."""
    print_section("Single Prediction")
    
    # Sample customer data
    customer_data = {
        "customer_id": "CUST001",
        "features": {
            "gender": "Male",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 12,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 70.35,
            "total_charges": 840.75
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def example_batch_prediction():
    """Example: Batch prediction."""
    print_section("Batch Prediction")
    
    # Sample batch data
    batch_data = {
        "customers": [
            {
                "customer_id": "CUST001",
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 70.35,
                "total_charges": 840.75
            },
            {
                "customer_id": "CUST002",
                "gender": "Female",
                "senior_citizen": 1,
                "partner": "No",
                "dependents": "No",
                "tenure": 48,
                "phone_service": "Yes",
                "multiple_lines": "Yes",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "Yes",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Two year",
                "paperless_billing": "No",
                "payment_method": "Bank transfer (automatic)",
                "monthly_charges": 85.50,
                "total_charges": 4104.00
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  CUSTOMER CHURN PREDICTION API - USAGE EXAMPLES")
    print("=" * 70)
    print("\nMake sure the API server is running:")
    print("  python serve.py")
    print("\nOr with Docker:")
    print("  docker-compose up")
    print("\n" + "=" * 70)
    
    try:
        # Run examples
        example_health_check()
        example_model_info()
        example_single_prediction()
        example_batch_prediction()
        
        print("\n" + "=" * 70)
        print("  All examples completed successfully!")
        print("=" * 70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to API server.")
        print("Please make sure the server is running:")
        print("  python serve.py")
        print("\n")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}\n")


if __name__ == "__main__":
    main()
