import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import precision_score, recall_score

# Paths
MODEL_PATH = Path("models/baseline_model.joblib")
PRODUCTION_BATCH_DIR = Path("data/production_batches")
BIAS_OUTPUT_DIR = Path("monitoring/bias_reports")

BIAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensitive attributes to monitor
SENSITIVE_FEATURES = ["gender", "SeniorCitizen", "Partner"]

# Load trained model
model = joblib.load(MODEL_PATH)


def compute_group_metrics(df, feature):
    """
    Computing precision and recall on sensitive features.
    """
    results = []

    for group_value in df[feature].unique():
        group_df = df[df[feature] == group_value]

        # Skipping very small groups (unstable metrics)
        if len(group_df) < 30:
            continue

        X = group_df.drop(columns=["Churn"])
        y_true = group_df["Churn"]

        y_pred = model.predict(X)

        precision = precision_score(y_true, y_pred, pos_label="Yes")
        recall = recall_score(y_true, y_pred, pos_label="Yes")

        results.append({
            "group": group_value,
            "count": len(group_df),
            "precision": precision,
            "recall": recall
        })

    return results



# Process production batches
for batch_file in sorted(PRODUCTION_BATCH_DIR.glob("production_batch_*.csv")):
    batch_df = pd.read_csv(batch_file)

    # Schema consistency
    for col in batch_df.select_dtypes(include=["object"]).columns:
        batch_df[col] = batch_df[col].astype(str)

    batch_bias_report = {
        "batch": batch_file.stem,
        "features": {}
    }

    for feature in SENSITIVE_FEATURES:
        batch_bias_report["features"][feature] = compute_group_metrics(
            batch_df, feature
        )

    output_path = BIAS_OUTPUT_DIR / f"{batch_file.stem}_bias.json"
    pd.Series(batch_bias_report).to_json(output_path, indent=2)

    print(f"Saved bias report: {output_path}")
