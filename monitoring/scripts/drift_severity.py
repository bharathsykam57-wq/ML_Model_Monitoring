import pandas as pd
from pathlib import Path

# Importing drift storage function
from store_drift_metrics import store_drift_metric

# Paths
REFERENCE_DATA_PATH = Path("data/reference/reference_data.csv")
PRODUCTION_BATCH_DIR = Path("data/production_batches")

# Drift severity thresholds
LOW_THRESHOLD = 0.05
MEDIUM_THRESHOLD = 0.15


def classify_drift(drift_score):
    """
    Converting a numeric drift score into a severity label.
    """
    if drift_score < LOW_THRESHOLD:
        return "LOW"
    elif drift_score < MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def compute_numeric_drift(reference, production):
    """
    Simple numeric drift: absolute mean difference.
    """
    return abs(production.mean() - reference.mean())


def main():
    # Loading reference data
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    # Identifying numeric features only
    numeric_features = reference_df.select_dtypes(exclude=["object"]).columns

    # Processing each production batch
    for batch_file in sorted(PRODUCTION_BATCH_DIR.glob("production_batch_*.csv")):
        batch_name = batch_file.stem
        production_df = pd.read_csv(batch_file)

        for feature in numeric_features:
            drift_score = compute_numeric_drift(
                reference_df[feature],
                production_df[feature]
            )

            drift_level = classify_drift(drift_score)

            # Store drift result
            store_drift_metric(
                batch=batch_name,
                feature=feature,
                drift_score=drift_score,
                drift_level=drift_level
            )

        print(f"Drift severity processed for {batch_name}")


if __name__ == "__main__":
    main()