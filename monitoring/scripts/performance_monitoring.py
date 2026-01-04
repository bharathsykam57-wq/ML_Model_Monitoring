import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from datetime import datetime

# Configuration
POSITIVE_LABEL = "Yes"

MODEL_PATH = Path("models/baseline_model.joblib")
PRODUCTION_BATCH_DIR = Path("data/production_batches")

# Snapshot report (overwritten each run)
SNAPSHOT_OUTPUT_PATH = Path(
    "monitoring/performance_reports/model_performance.csv"
)

# Time-series metrics store (append-only, fixed schema)
METRICS_STORE_PATH = Path(
    "monitoring/metrics_store/performance_metrics.csv"
)

SNAPSHOT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Loading trained model
model = joblib.load(MODEL_PATH)

snapshot_records = []

# Processing each production batch
for batch_file in sorted(PRODUCTION_BATCH_DIR.glob("production_batch_*.csv")):
    batch_df = pd.read_csv(batch_file)

    # Enforcing schema consistency (matches training)
    for col in batch_df.select_dtypes(include=["object"]).columns:
        batch_df[col] = batch_df[col].astype(str)

    batch_size = len(batch_df)

    X = batch_df.drop(columns=["Churn"])
    y_true = batch_df["Churn"]

    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    precision = precision_score(
        y_true, y_pred, pos_label=POSITIVE_LABEL
    )
    recall = recall_score(
        y_true, y_pred, pos_label=POSITIVE_LABEL
    )
    roc_auc = roc_auc_score(
        (y_true == POSITIVE_LABEL).astype(int),
        y_pred_proba
    )

    
    # Snapshot report (overwritten each run)
    snapshot_records.append({
        "batch": batch_file.stem,
        "batch_size": batch_size,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    })

    
    # Time-series metrics store
    # FIXED SCHEMA (DO NOT CHANGE)
    metrics_row = pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "batch": batch_file.stem,
        "batch_size": batch_size,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }])

    if METRICS_STORE_PATH.exists():
        metrics_row.to_csv(
            METRICS_STORE_PATH,
            mode="a",
            header=False,
            index=False
        )
    else:
        metrics_row.to_csv(
            METRICS_STORE_PATH,
            index=False
        )


# Saving snapshot report
snapshot_df = pd.DataFrame(snapshot_records)
snapshot_df.to_csv(SNAPSHOT_OUTPUT_PATH, index=False)

print("Performance monitoring metrics generated successfully.")