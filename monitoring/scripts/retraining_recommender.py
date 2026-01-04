"""
Translating monitoring signals (performance, drift, bias)
into a concrete human decision: retrain or not. 
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

# Paths to stored metrics
PERFORMANCE_PATH = Path("monitoring/metrics_store/performance_metrics.csv")
DRIFT_PATH = Path("monitoring/metrics_store/drift_metrics.csv")
BIAS_PATH = Path("monitoring/metrics_store/bias_metrics.csv")

# Paths where decisions will be saved
DECISION_PATH = Path("monitoring/decisions/retraining_decisions.csv")
DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)

# Thresholds (business policy)
MIN_PRECISION = 0.60
MAX_ALLOWED_HIGH_DRIFT = 2
MAX_BIAS_GAP = 0.15

# Ensuring metric files exist
if not PERFORMANCE_PATH.exists():
    raise RuntimeError("Performance metrics not found")

if not DRIFT_PATH.exists():
    raise RuntimeError("Drift metrics not found")

if not BIAS_PATH.exists():
    raise RuntimeError("Bias metrics not found")


def recommend_action(batch_name: str):
    """
    Evaluating all monitoring signals for a batch
    and return an action + reasons.
    """

    reasons = []
    action = "NO_ACTION"

    # Load latest performance metrics
    perf_df = pd.read_csv(PERFORMANCE_PATH)
    batch_perf = perf_df[perf_df["batch"] == batch_name].iloc[-1]

    if batch_perf["precision"] < MIN_PRECISION:
        reasons.append(
            f"Precision below threshold ({batch_perf['precision']:.3f} < {MIN_PRECISION})"
        )
        action = "RETRAIN"

    # Load drift metrics
    drift_df = pd.read_csv(DRIFT_PATH)
    batch_drift = drift_df[
        (drift_df["batch"] == batch_name) &
        (drift_df["drift_level"] == "HIGH")
    ]

    if len(batch_drift) >= MAX_ALLOWED_HIGH_DRIFT:
        reasons.append(
            f"High drift detected in {len(batch_drift)} features"
        )
        action = "RETRAIN"

    # Load bias metrics (recall gaps)
    bias_df = pd.read_csv(BIAS_PATH)
    batch_bias = bias_df[bias_df["batch"] == batch_name]

    # Check recall disparity within same sensitive feature
    for feature in batch_bias["feature"].unique():
        feature_df = batch_bias[batch_bias["feature"] == feature]
        gap = feature_df["recall"].max() - feature_df["recall"].min()

        if gap > MAX_BIAS_GAP:
            reasons.append(
                f"Bias detected in {feature} (recall gap {gap:.2f})"
            )
            if action != "RETRAIN":
                action = "ESCALATE_FAIRNESS"

    # Save decision record
    decision_record = {
        "timestamp": datetime.utcnow(),
        "batch": batch_name,
        "action": action,
        "reasons": "; ".join(reasons) if reasons else "All metrics stable"
    }

    decision_df = pd.DataFrame([decision_record])

    if DECISION_PATH.exists():
        decision_df.to_csv(DECISION_PATH, mode="a", header=False, index=False)
    else:
        decision_df.to_csv(DECISION_PATH, index=False)

    return decision_record

# Action precedence: 
# RETRAIN > ESCALATE_FAIRNESS > NO_ACTION 
# Run for latest batch only
if __name__ == "__main__":
    perf_df = pd.read_csv(PERFORMANCE_PATH)
    latest_batch = perf_df["batch"].iloc[-1]

    decision = recommend_action(latest_batch)

    print("\nRETRAINING DECISION")
    print("------------------")
    print(f"Batch: {decision['batch']}")
    print(f"Action: {decision['action']}")
    print("Reasons:")
    print(f"- {decision['reasons']}")