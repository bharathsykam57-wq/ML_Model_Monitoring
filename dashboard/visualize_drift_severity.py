"""
Visualize overall drift severity across production batches.
This answers the question:
"Is the system getting worse over time?"
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DRIFT_METRICS_PATH = Path("monitoring/metrics_store/drift_metrics.csv")
OUTPUT_PATH = Path("dashboard/drift_severity_composition.png")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Loading data
df = pd.read_csv(DRIFT_METRICS_PATH)

# Counts severity levels per batch
severity_counts = (
    df.groupby(["batch", "drift_level"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

# Ensuring consistent known order
severity_counts = severity_counts[["LOW", "MEDIUM", "HIGH"]]


# Plot stacked bars
severity_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5),
    color=["#2ecc71", "#f1c40f", "#e74c3c"]
)

plt.xlabel("Production Batch")
plt.ylabel("Number of Features")
plt.title("Drift Severity Composition per Batch")
plt.legend(title="Drift Level")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()

print(f"Saved drift severity composition plot to {OUTPUT_PATH}")