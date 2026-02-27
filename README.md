# ML Fraud Detection System Overview

Welcome to the **ML-Based Fraud Detection System** for crypto and financial transactions.

## Setup Instructions

If you haven't run the initial setup, please run the following commands in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the 50,000 transaction dataset with a 3% fraud ratio
python src/dataset_generator.py

# 3. Train the Machine Learning models (LogReg, RandomForest, XGBoost) and save best models
python src/model_trainer.py

# 4. Start the interactive User Interface Dashboard
python -m streamlit run app.py
```

## Dashboard Features
Once the Streamlit server is running, navigate to `http://localhost:8501` to use the intelligent dashboard.

1. **Overview & Insights**: Understand dataset distribution and visually analyze `Fraud vs Legitimate` behavior across multiple transaction types (e.g. `CASH_OUT`, `TRANSFER`, `PAYMENT`).
2. **Model Performance**: Compare multiple top-tier models evaluated on the generated test set. The models are robust and handled class-imbalance using the `SMOTE` oversampling technique.
   * *Best Model usually evaluated by F1 Score and Recall will be auto-selected.*
3. **Predict Fraud**: Enter custom transaction parameters interactively (like Amount, Balances, and Time of Day) and watch the XGBoost/RandomForest engine evaluate the fraud risk score in real-time.

Enjoy interacting with the automated predictive dashboard!
