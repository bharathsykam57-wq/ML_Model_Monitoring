"""
Alert Engine for ML Model Monitoring

This module reads stored monitoring metrics (performance, drift, bias)
and raises alerts when predefined thresholds are violated.

Alerts are printed to console (can later be extended to email/Slack).
"""

import pandas as pd
from pathlib import Path

# Paths to metric stores
PERFORMANCE_METRICS_PATH = Path("monitoring/metrics_store/performance_metrics.csv")
BIAS_METRICS_PATH = Path("monitoring/metrics_store/bias_metrics.csv")
DRIFT_METRICS_PATH = Path("monitoring/metrics_store/drift_metrics.csv")


# Alert thresholds 
PERFORMANCE_THRESHOLDS = {
    "precision_min": 0.60,
    "recall_min": 0.55,
    "roc_auc_min": 0.80,
}

DRIFT_THRESHOLDS = {
    "high_drift_count": 2  # number of HIGH drift features allowed per batch
}

BIAS_THRESHOLDS = {
    "max_recall_gap": 0.15  # max allowed difference between groups
}


# Performance alerts
def check_performance_alerts():
    if not PERFORMANCE_METRICS_PATH.exists():
        print("Performance metrics file not found.")
        return

    df = pd.read_csv(PERFORMANCE_METRICS_PATH)

    latest = df.sort_values("timestamp").iloc[-1]

    alerts = []

    if latest["precision"] < PERFORMANCE_THRESHOLDS["precision_min"]:
        alerts.append(f"Low precision: {latest['precision']:.3f}")

    if latest["recall"] < PERFORMANCE_THRESHOLDS["recall_min"]:
        alerts.append(f"Low recall: {latest['recall']:.3f}")

    if latest["roc_auc"] < PERFORMANCE_THRESHOLDS["roc_auc_min"]:
        alerts.append(f"Low ROC-AUC: {latest['roc_auc']:.3f}")

    if alerts:
        print(f" PERFORMANCE ALERT ({latest['batch']})")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("Performance within acceptable thresholds.")


# Drift alerts
def check_drift_alerts():
    if not DRIFT_METRICS_PATH.exists():
        print("Drift metrics file not found.")
        return

    df = pd.read_csv(DRIFT_METRICS_PATH)

    latest_batch = df.sort_values("timestamp")["batch"].iloc[-1]
    batch_df = df[df["batch"] == latest_batch]

    high_drift_features = batch_df[batch_df["drift_level"] == "HIGH"]

    if len(high_drift_features) >= DRIFT_THRESHOLDS["high_drift_count"]:
        print(f" DRIFT ALERT ({latest_batch})")
        print(f"  High drift detected in {len(high_drift_features)} features:")
        for feature in high_drift_features["feature"]:
            print(f"  - {feature}")
    else:
        print(" Drift levels acceptable.")



# Bias alerts
def check_bias_alerts():
    if not BIAS_METRICS_PATH.exists():
        print("Bias metrics file not found.")
        return

    df = pd.read_csv(BIAS_METRICS_PATH)

    latest_batch = df.sort_values("timestamp")["batch"].iloc[-1]
    batch_df = df[df["batch"] == latest_batch]

    alerts_triggered = False

    for feature in batch_df["feature"].unique():
        feature_df = batch_df[batch_df["feature"] == feature]

        recall_gap = feature_df["recall"].max() - feature_df["recall"].min()

        if recall_gap > BIAS_THRESHOLDS["max_recall_gap"]:
            alerts_triggered = True
            print(f" BIAS ALERT ({latest_batch})")
            print(f"  Feature: {feature}")
            print(f"  Recall gap: {recall_gap:.3f}")

    if not alerts_triggered:
        print("No significant bias detected.")


# Main execution
def run_all_alerts():
    print("\n RUNNING ALERT ENGINE")
    check_performance_alerts()
    check_drift_alerts()
    check_bias_alerts()
    print("\n ALERT CHECK COMPLETE")


if __name__ == "__main__":
    run_all_alerts()