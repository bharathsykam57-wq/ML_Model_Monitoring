from pathlib import Path
import pandas as pd
from datetime import datetime

STORE_PATH = Path("monitoring/metrics_store/performance_metrics.csv")
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

def store_performance_metrics(batch, precision, recall, roc_auc):
    record = {
        "timestamp": datetime.utcnow(),
        "batch": batch,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }

    df = pd.DataFrame([record])

    if STORE_PATH.exists():
        df.to_csv(STORE_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(STORE_PATH, index=False)