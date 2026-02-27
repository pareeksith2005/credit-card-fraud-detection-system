import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess_data(data_path="data/transactions.csv", model_dir="models/"):
    """
    Loads data, handles categorical features, scales numerical data,
    and applies SMOTE to handle class imbalance.
    """
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['is_fraud', 'step']) # drop step as it's just a time counter
    y = df['is_fraud']
    
    # Handle categorical variables (Encode 'type')
    le = LabelEncoder()
    X['type'] = le.fit_transform(X['type'])
    
    # Train-Test Split (before SMOTE to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to the training set ONLY
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))
    
    print(f"Original Training Class Distribution:\n{y_train.value_counts()}")
    print(f"Resampled Training Class Distribution:\n{y_train_resampled.value_counts()}")
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, X.columns

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cols = load_and_preprocess_data()
