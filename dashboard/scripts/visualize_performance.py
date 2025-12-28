import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to performance metrics
PERFORMANCE_PATH = Path("monitoring/performance_reports/model_performance.csv")

# Loading performance data
df = pd.read_csv(PERFORMANCE_PATH)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Loading performance metrics
PERF_PATH = Path("monitoring/performance_reports/model_performance.csv")
df = pd.read_csv(PERF_PATH)

# Sort by batch index (important for correct trend)
df["batch_index"] = df["batch"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("batch_index")

# Plotting performance trends
plt.figure(figsize=(10, 5))
plt.plot(df["batch"], df["precision"], marker="o", label="Precision")
plt.plot(df["batch"], df["recall"], marker="o", label="Recall")
plt.plot(df["batch"], df["roc_auc"], marker="o", label="ROC-AUC")

plt.title("Model Performance Over Batches")
plt.xlabel("Production Batch")
plt.ylabel("Metric Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dashboard/performance_over_time.png", dpi=300, bbox_inches="tight")
plt.show()