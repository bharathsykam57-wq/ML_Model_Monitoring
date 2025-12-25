import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BIAS_DIR = Path("monitoring/bias_reports")

records = []

for file in sorted(BIAS_DIR.glob("*_bias.json")):
    with open(file) as f:
        data = json.load(f)

    batch = data["batch"]

    for feature, groups in data["features"].items():
        for g in groups:
            records.append({
                "batch": batch,
                "feature": feature,
                "group": g["group"],
                "recall": g["recall"]
            })

df = pd.DataFrame(records)

# Example: visualize SeniorCitizen recall gap
subset = df[df["feature"] == "SeniorCitizen"]

plt.figure(figsize=(10, 5))
for group in subset["group"].unique():
    grp_df = subset[subset["group"] == group]
    plt.plot(grp_df.index, grp_df["recall"], marker="o", label=f"Group {group}")

plt.title("Recall by Group (SeniorCitizen)")
plt.xlabel("Time")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("dashboard/bias_recall_gap.png")
plt.show()
