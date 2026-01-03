"""
Computes group-wise recall for sensitive attributes
and stores results as a time-series CSV for alerting
and auditing.
"""

import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
from sklearn.metrics import recall_score

# Paths
MODEL_PATH = Path("models/baseline_model.joblib")
PRODUCTION_BATCH_DIR = Path("data/production_batches")
BIAS_METRICS_PATH = Path("monitoring/metrics_store/bias_metrics.csv")

BIAS_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configuration
SENSITIVE_FEATURES = ["gender", "SeniorCitizen", "Partner"]
MIN_GROUP_SIZE = 30

# Loading model
model = joblib.load(MODEL_PATH)

records = []

# Processing production batches
for batch_file in sorted(PRODUCTION_BATCH_DIR.glob("production_batch_*.csv")):
    df = pd.read_csv(batch_file)

    # Schema consistency (must be same training)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    X_all = df.drop(columns=["Churn"])
    y_all = df["Churn"]

    y_pred_all = model.predict(X_all)

    # Group-wise evaluation
    for feature in SENSITIVE_FEATURES:
        for group_value in df[feature].unique():

            mask = df[feature] == group_value
            group_size = mask.sum()

            if group_size < MIN_GROUP_SIZE:
                continue

            recall = recall_score(
                y_all[mask],
                y_pred_all[mask],
                pos_label="Yes"
            )

            records.append({
                "timestamp": datetime.utcnow(),
                "batch": batch_file.stem,
                "feature": feature,
                "group": str(group_value),
                "group_size": group_size,
                "recall": recall
            })

# Persist metrics
bias_df = pd.DataFrame(records)

if not bias_df.empty:
    if BIAS_METRICS_PATH.exists():
        bias_df.to_csv(BIAS_METRICS_PATH, mode="a", header=False, index=False)
    else:
        bias_df.to_csv(BIAS_METRICS_PATH, index=False)

print("Bias & fairness monitoring completed successfully.")