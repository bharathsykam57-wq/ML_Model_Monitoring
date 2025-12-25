import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to performance metrics
PERFORMANCE_PATH = Path("monitoring/performance_reports/model_performance.csv")

# Loading performance data
df = pd.read_csv(PERFORMANCE_PATH)

# Sorting by batch index to ensure time order
df["batch_index"] = df["batch"].str.extract(r"(\d+)").astype(int)
df = df.sort_values("batch_index")

# Plot metrics
plt.figure(figsize=(10, 6))
plt.plot(df["batch_index"], df["roc_auc"], label="ROC-AUC", marker="o")
plt.plot(df["batch_index"], df["precision"], label="Precision", marker="o")
plt.plot(df["batch_index"], df["recall"], label="Recall", marker="o")

plt.title("Model Performance Over Production Batches")
plt.xlabel("Production Batch")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("dashboard/performance_trends.png")
plt.show()
