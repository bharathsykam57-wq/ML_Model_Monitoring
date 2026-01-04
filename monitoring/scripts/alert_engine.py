"""
Alert Engine for ML Model Monitoring

Reads stored monitoring metrics (performance, drift, bias)
and raises alerts when predefined thresholds are violated.
"""

import pandas as pd
from pathlib import Path

# Metric store paths
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
    "max_high_drift_features": 2
}

BIAS_THRESHOLDS = {
    "max_recall_gap": 0.15
}


def check_performance_alerts():
    if not PERFORMANCE_METRICS_PATH.exists():
        print("Performance metrics file not found.")
        return

    df = pd.read_csv(PERFORMANCE_METRICS_PATH)
    if df.empty:
        print("Performance metrics file is empty.")
        return

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


def check_drift_alerts():
    if not DRIFT_METRICS_PATH.exists():
        print("Drift metrics file not found.")
        return

    df = pd.read_csv(DRIFT_METRICS_PATH)
    if df.empty:
        print("Drift metrics file is empty.")
        return

    latest_batch = df.sort_values("timestamp")["batch"].iloc[-1]
    batch_df = df[df["batch"] == latest_batch]

    high_drift_features = batch_df[batch_df["drift_level"] == "HIGH"]

    if len(high_drift_features) > DRIFT_THRESHOLDS["max_high_drift_features"]:
        print(f" DRIFT ALERT ({latest_batch})")
        print(f"  High drift detected in {len(high_drift_features)} features:")
        for feature in high_drift_features["feature"]:
            print(f"  - {feature}")
    else:
        print("Drift levels acceptable.")


def check_bias_alerts():
    if not BIAS_METRICS_PATH.exists():
        print("Bias metrics file not found.")
        return

    df = pd.read_csv(BIAS_METRICS_PATH)
    if df.empty:
        print("Bias metrics file is empty.")
        return

    latest_batch = df.sort_values("timestamp")["batch"].iloc[-1]
    batch_df = df[df["batch"] == latest_batch]

    alerts_triggered = False

    for feature in batch_df["feature"].unique():
        feature_df = batch_df[batch_df["feature"] == feature]

        # Skip if only one group present
        if feature_df["group"].nunique() < 2:
            continue

        recall_gap = feature_df["recall"].max() - feature_df["recall"].min()

        if recall_gap > BIAS_THRESHOLDS["max_recall_gap"]:
            if not alerts_triggered:
                print(f" BIAS ALERT ({latest_batch})")
                alerts_triggered = True

            print(f"  Feature: {feature}")
            print(f"  Recall gap: {recall_gap:.3f}")

    if not alerts_triggered:
        print("No significant bias detected.")


def run_all_alerts():
    print("\n RUNNING ALERT ENGINE")
    check_performance_alerts()
    check_drift_alerts()
    check_bias_alerts()
    print("\n ALERT CHECK COMPLETE")


if __name__ == "__main__":
    run_all_alerts()