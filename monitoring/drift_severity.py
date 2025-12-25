import json
from pathlib import Path

DRIFT_REPORT_DIR = Path("monitoring/drift_reports")
SEVERITY_OUTPUT_DIR = Path("monitoring/drift_severity")
SEVERITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def numerical_severity(mean_diff, reference_std):
    """
    Classifies drift severity for numerical features.

    normalizing the mean difference by the reference standard deviation
    so that severity is scale-invariant.
    """
    ratio = abs(mean_diff) / (reference_std + 1e-8)

    if ratio < 0.05:
        return "low"
    elif ratio < 0.15:
        return "medium"
    else:
        return "high"


def categorical_severity(freq_diff):
    """
    Classifying drift severity for categorical features
    based on absolute frequency change.
    """
    abs_diff = abs(freq_diff)

    if abs_diff < 0.05:
        return "low"
    elif abs_diff < 0.15:
        return "medium"
    else:
        return "high"


def main():
    # Iterating through all drift reports 
    for drift_file in DRIFT_REPORT_DIR.glob("*_drift.json"):
        with open(drift_file) as f:
            drift_data = json.load(f)

        # Initializing batch-level severity summary
        severity_summary = {
            "batch": drift_file.stem,
            "features": {},
            "overall_status": "low"
        }

        # Processing each feature's drift statistics
        for feature, stats in drift_data.items():

            # Numerical feature: identified by presence of mean_difference
            if "mean_difference" in stats:
                severity = numerical_severity(
                    stats["mean_difference"],
                    stats["reference_std"]
                )

            # Categorical feature: stats is a dictionary of category frequencies
            else:
                severities = [
                    categorical_severity(v["freq_difference"])
                    for v in stats.values()
                ]

                # Taking worst-case severity across categories
                severity = (
                    "high" if "high" in severities else
                    "medium" if "medium" in severities else
                    "low"
                )

            # Storing feature-level severity
            severity_summary["features"][feature] = severity

            # Updating overall batch status (worst-case logic)
            if severity == "high":
                severity_summary["overall_status"] = "high"
            elif severity == "medium" and severity_summary["overall_status"] != "high":
                severity_summary["overall_status"] = "medium"

        # Saving severity report
        output_path = SEVERITY_OUTPUT_DIR / f"{drift_file.stem}_severity.json"
        with open(output_path, "w") as f:
            json.dump(severity_summary, f, indent=2)

        print(f"Saved severity report: {output_path}")


if __name__ == "__main__":
    main()
