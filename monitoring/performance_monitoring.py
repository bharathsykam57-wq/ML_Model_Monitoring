import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Paths

# Trained baseline model
MODEL_PATH = Path("models/baseline_model.joblib")

# Simulated production data batches
PRODUCTION_BATCH_DIR = Path("data/production_batches")

# Output file for performance monitoring results
PERFORMANCE_OUTPUT_PATH = Path("monitoring/performance_reports/model_performance.csv")

# Loading trained model
model = joblib.load(MODEL_PATH)

# Ensure output directory exists
PERFORMANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

performance_records = []

# Processing each production batch
for batch_file in sorted(PRODUCTION_BATCH_DIR.glob("production_batch_*.csv")):
    batch_df = pd.read_csv(batch_file)

    # Ensure schema compatibility (categoricals as strings)
    for col in batch_df.select_dtypes(include=["object"]).columns:
        batch_df[col] = batch_df[col].astype(str)

    # Separating features and target
    X_batch = batch_df.drop(columns=["Churn"])
    y_true = batch_df["Churn"]

    # Model predictions
    y_pred = model.predict(X_batch)
    y_pred_proba = model.predict_proba(X_batch)[:, 1]

    # Performance metrics
    precision = precision_score(y_true, y_pred, pos_label="Yes")
    recall = recall_score(y_true, y_pred, pos_label="Yes")

    # ROC AUC requires binary labels
    y_true_binary = (y_true == "Yes").astype(int)
    roc_auc = roc_auc_score(y_true_binary, y_pred_proba)

    performance_records.append({
        "batch": batch_file.stem,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    })

# performance report
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv(PERFORMANCE_OUTPUT_PATH, index=False)

print("Performance monitoring report generated successfully.")
