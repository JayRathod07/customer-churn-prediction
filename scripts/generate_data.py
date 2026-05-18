"""Generate synthetic customer churn data for demonstration purposes."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_customer_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic customer churn data.
    
    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic customer data
    """
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    
    # Generate demographic features
    gender = np.random.choice(['Male', 'Female'], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.52, 0.48])
    dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])
    
    # Generate tenure (months with company) - skewed distribution
    tenure = np.random.gamma(shape=2, scale=15, size=n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)  # Cap at 6 years
    
    # Generate service features
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10])
    
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.42, 0.58]))
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.34, 0.44, 0.22])
    
    # Internet-dependent services
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for internet in internet_service:
        if internet == 'No':
            online_security.append('No internet service')
            online_backup.append('No internet service')
            device_protection.append('No internet service')
            tech_support.append('No internet service')
            streaming_tv.append('No internet service')
            streaming_movies.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.29, 0.71]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.34, 0.66]))
            device_protection.append(np.random.choice(['Yes', 'No'], p=[0.34, 0.66]))
            tech_support.append(np.random.choice(['Yes', 'No'], p=[0.29, 0.71]))
            streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.38, 0.62]))
            streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.39, 0.61]))
    
    # Contract type - influences churn
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                size=n_samples, p=[0.55, 0.21, 0.24])
    
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.59, 0.41])
    
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        size=n_samples, p=[0.34, 0.15, 0.22, 0.29]
    )
    
    # Generate charges
    # Monthly charges depend on services
    base_charge = 20
    monthly_charges = []
    
    for i in range(n_samples):
        charge = base_charge
        
        if phone_service[i] == 'Yes':
            charge += 10
        if multiple_lines[i] == 'Yes':
            charge += 10
        if internet_service[i] == 'DSL':
            charge += 25
        elif internet_service[i] == 'Fiber optic':
            charge += 45
        if online_security[i] == 'Yes':
            charge += 5
        if online_backup[i] == 'Yes':
            charge += 5
        if device_protection[i] == 'Yes':
            charge += 5
        if tech_support[i] == 'Yes':
            charge += 5
        if streaming_tv[i] == 'Yes':
            charge += 10
        if streaming_movies[i] == 'Yes':
            charge += 10
        
        # Add some random variation
        charge += np.random.normal(0, 5)
        monthly_charges.append(max(charge, 18.25))  # Minimum charge
    
    monthly_charges = np.array(monthly_charges)
    
    # Total charges = monthly charges * tenure (with some variation)
    total_charges = monthly_charges * tenure
    total_charges = total_charges + np.random.normal(0, 50, size=n_samples)
    total_charges = np.maximum(total_charges, 0)  # No negative charges
    
    # Generate churn label with realistic patterns
    churn_probability = []
    
    for i in range(n_samples):
        prob = 0.2  # Base churn rate
        
        # Tenure effect (longer tenure = less churn)
        if tenure[i] < 6:
            prob += 0.3
        elif tenure[i] < 12:
            prob += 0.15
        elif tenure[i] > 48:
            prob -= 0.15
        
        # Contract effect
        if contract[i] == 'Month-to-month':
            prob += 0.25
        elif contract[i] == 'Two year':
            prob -= 0.20
        
        # Internet service effect
        if internet_service[i] == 'Fiber optic':
            prob += 0.10
        
        # Payment method effect
        if payment_method[i] == 'Electronic check':
            prob += 0.10
        
        # Monthly charges effect (higher charges = more churn)
        if monthly_charges[i] > 80:
            prob += 0.15
        
        # Senior citizen effect
        if senior_citizen[i] == 1:
            prob += 0.05
        
        # No dependents effect
        if dependents[i] == 'No':
            prob += 0.05
        
        churn_probability.append(np.clip(prob, 0, 1))
    
    churn = np.random.binomial(1, churn_probability)
    churn = ['Yes' if c == 1 else 'No' for c in churn]
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'churn': churn
    })
    
    return df


def main():
    """Main function to generate and save synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic customer churn data')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/customer_churn.csv', help='Output file path')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} synthetic customer records...")
    df = generate_customer_data(n_samples=args.n_samples, random_state=args.random_state)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Data saved to {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Churn rate: {(df['churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    print(f"  Features: {len(df.columns)}")
    print(f"\nChurn distribution:")
    print(df['churn'].value_counts())


if __name__ == '__main__':
    main()
