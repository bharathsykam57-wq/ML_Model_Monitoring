import json
from pathlib import Path
import matplotlib.pyplot as plt

SEVERITY_DIR = Path("monitoring/drift_severity")

batch_indices = []
severity_scores = []

severity_map = {"low": 0, "medium": 1, "high": 2}

for file in sorted(SEVERITY_DIR.glob("*_severity.json")):
    with open(file) as f:
        data = json.load(f)

    batch_index = int(data["batch"].split("_")[-2])
    severity = severity_map[data["overall_status"]]

    batch_indices.append(batch_index)
    severity_scores.append(severity)

plt.figure(figsize=(8, 4))
plt.plot(batch_indices, severity_scores, marker="o")
plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
plt.xlabel("Production Batch")
plt.ylabel("Drift Severity")
plt.title("Overall Drift Severity Over Time")
plt.grid(True)

plt.tight_layout()
plt.savefig("dashboard/drift_severity_trend.png")
plt.show()
