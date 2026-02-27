import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import json
import os

def train_and_evaluate_models():
    """
    Trains multiple ML models and evaluates them.
    Saves the models and their performance metrics.
    """
    # Import locally from src.preprocessing to avoid path issues
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import load_and_preprocess_data

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data(
        data_path="data/transactions.csv", model_dir="models/"
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }

    results = {}
    best_f1 = 0
    best_model_name = ""

    os.makedirs("models", exist_ok=True)
    
    # Save feature names
    joblib.dump(list(feature_cols), "models/feature_columns.pkl")

    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1, 4),
            'ROC AUC': round(roc_auc, 4)
        }
        
        print(f"{name} Results - F1: {f1:.4f} | Recall: {recall:.4f}")
        
        # Save model
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")
    
    # Mark the best model
    results['Best_Model'] = best_model_name
    
    # Save results to a JSON file
    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Models and metrics saved successfully in 'models/' directory.")

if __name__ == "__main__":
    train_and_evaluate_models()
