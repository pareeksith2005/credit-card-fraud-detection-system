import pandas as pd
import numpy as np
import os

def generate_transactions(n_samples=50000, fraud_ratio=0.03, output_path="data/transactions.csv"):
    """
    Generate synthetic dataset of financial transactions including fraud cases.
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legit transactions distribution
    legit_amounts = np.random.lognormal(mean=3, sigma=1.5, size=n_legit)
    
    # Fraud transactions generally higher amounts, potentially more extreme
    fraud_amounts = np.random.lognormal(mean=7, sigma=1.0, size=n_fraud)
    
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    
    # Types of transactions
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    legit_types = np.random.choice(types, size=n_legit, p=[0.35, 0.15, 0.25, 0.05, 0.20])
    
    # Fraud often happens in transfers or cash_outs
    fraud_types = np.random.choice(['TRANSFER', 'CASH_OUT'], size=n_fraud, p=[0.5, 0.5])
    
    transaction_types = np.concatenate([legit_types, fraud_types])
    
    # Balance before transaction
    legit_oldbalanceOrg = np.random.lognormal(mean=8, sigma=2, size=n_legit)
    fraud_oldbalanceOrg = np.random.lognormal(mean=9, sigma=1.5, size=n_fraud)
    oldbalanceOrg = np.concatenate([legit_oldbalanceOrg, fraud_oldbalanceOrg])
    
    # Compute newbalanceOrig (Simplified logic)
    newbalanceOrig = np.maximum(0, oldbalanceOrg - amounts)
    
    # Dest accounts (Simplified)
    oldbalanceDest = np.random.lognormal(mean=9, sigma=2, size=n_samples)
    newbalanceDest = oldbalanceDest + amounts
    
    # Time (Hours in a day 0-23)
    legit_time = np.random.randint(6, 23, size=n_legit) # Most legit during day
    fraud_time = np.random.randint(0, 24, size=n_fraud)   # Fraud at any time
    time_of_day = np.concatenate([legit_time, fraud_time])
    
    # Labels
    is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    
    # Shuffle dataset
    idx = np.random.permutation(n_samples)
    
    df = pd.DataFrame({
        'step': np.random.randint(1, 744, size=n_samples), # Represents hours over a month
        'type': transaction_types[idx],
        'amount': amounts[idx],
        'oldbalanceOrg': oldbalanceOrg[idx],
        'newbalanceOrig': newbalanceOrig[idx],
        'oldbalanceDest': oldbalanceDest[idx],
        'newbalanceDest': newbalanceDest[idx],
        'time_of_day': time_of_day[idx],
        'is_fraud': is_fraud[idx].astype(int)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {n_samples} samples at {output_path}")
    print(f"Fraud count: {n_fraud} ({fraud_ratio*100:.2f}%)")
    
if __name__ == "__main__":
    generate_transactions()
