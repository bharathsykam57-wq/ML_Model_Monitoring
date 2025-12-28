from pathlib import Path
import pandas as pd
from datetime import datetime

# Path where drift metrics will be stored
STORE_PATH = Path("monitoring/metrics_store/drift_metrics.csv")

# Ensuring the directory exists
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def store_drift_metric(batch, feature, drift_score, drift_level):
    """
    Store one drift metric record.

    Parameters
    batch : str
        Production batch name (e.g., production_batch_0)
    feature : str
        Feature name (e.g., MonthlyCharges)
    drift_score : float
        Numeric drift score (mean diff, PSI, etc.)
    drift_level : str
        Severity label: LOW / MEDIUM / HIGH
    """

    record = {
        "timestamp": datetime.utcnow(),
        "batch": batch,
        "feature": feature,
        "drift_score": drift_score,
        "drift_level": drift_level,
    }

    df = pd.DataFrame([record])

    # Append if file exists, else create new
    if STORE_PATH.exists():
        df.to_csv(STORE_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(STORE_PATH, index=False)