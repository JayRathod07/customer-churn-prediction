"""
API Server Script

Start the FastAPI server for churn prediction.

Author: Jay Rathod
GitHub: jayRathod07
Email: jayrathod121005@gmail.com
"""

import argparse
from src.api.app import start_server


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description='Start Customer Churn Prediction API Server')
    parser.add_argument('--host', type=str, default=None, help='Host address (default from config)')
    parser.add_argument('--port', type=int, default=None, help='Port number (default from config)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  CUSTOMER CHURN PREDICTION API SERVER")
    print("=" * 70)
    print(f"\nStarting server...")
    print(f"Host: {args.host or 'from config'}")
    print(f"Port: {args.port or 'from config'}")
    print(f"Reload: {args.reload}")
    print(f"\nAPI Documentation: http://localhost:{args.port or 8000}/docs")
    print("=" * 70 + "\n")
    
    start_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
