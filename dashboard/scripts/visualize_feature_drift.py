import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading drift metrics
df = pd.read_csv("monitoring/metrics_store/drift_metrics.csv")

# Map drift levels to numeric scale
severity_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
df["severity_num"] = df["drift_level"].map(severity_map)

# Pivot for heatmap
heatmap_df = df.pivot(
    index="feature",
    columns="batch",
    values="severity_num"
)

plt.figure(figsize=(10, 4))
sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="RdYlGn_r",
    cbar_kws={"label": "Drift Severity"}
)

plt.title("Feature-Level Drift Severity Heatmap")
plt.xlabel("Production Batch")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("dashboard/feature_drift_heatmap.png")
plt.show()