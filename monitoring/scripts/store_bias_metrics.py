from pathlib import Path
import pandas as pd
from datetime import datetime

STORE_PATH = Path("monitoring/metrics_store/bias_metrics.csv")
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

def store_bias_metrics(batch, feature, group, group_size, recall):
    record = {
        "timestamp": datetime.utcnow(),
        "batch": batch,
        "feature": feature,
        "group": group,
        "group_size": group_size,
        "recall": recall,
    }

    df = pd.DataFrame([record])

    if STORE_PATH.exists():
        df.to_csv(STORE_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(STORE_PATH, index=False)