"""
This module computes feature-level drift signals by comparing
reference data against production batches.

- This file logs raw diagnostic signals (means, stds, missing rates, PSI).
- Drift decisions and thresholds are applied downstream
  in drift_severity.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Paths
REFERENCE_PATH = Path("data/reference/reference_data.csv")
PRODUCTION_DIR = Path("data/production_batches")
OUTPUT_DIR = Path("monitoring/drift_reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Loading reference data
reference_df = pd.read_csv(REFERENCE_PATH)

categorical_features = reference_df.select_dtypes(include=["object"]).columns
numerical_features = reference_df.select_dtypes(exclude=["object"]).columns


# Utility: PSI computation
def compute_psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """
    Computing Population Stability Index (PSI)
    for numerical features using equal-width bins.
    """

    ref = ref.dropna()
    prod = prod.dropna()

    if ref.empty or prod.empty:
        return 0.0

    breakpoints = np.linspace(
        min(ref.min(), prod.min()),
        max(ref.max(), prod.max()),
        bins + 1,
    )

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    prod_counts, _ = np.histogram(prod, bins=breakpoints)

    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    prod_dist = prod_counts / max(prod_counts.sum(), 1)

    psi = np.sum(
        (prod_dist - ref_dist)
        * np.log((prod_dist + 1e-6) / (ref_dist + 1e-6))
    )

    return float(psi)


def compute_categorical_psi(ref: pd.Series, prod: pd.Series) -> float:
    """
    Computing PSI for categorical features.
    """

    ref_dist = ref.value_counts(normalize=True)
    prod_dist = prod.value_counts(normalize=True)

    psi = 0.0
    all_categories = set(ref_dist.index).union(set(prod_dist.index))

    for category in all_categories:
        r = ref_dist.get(category, 1e-6)
        p = prod_dist.get(category, 1e-6)
        psi += (p - r) * np.log(p / r)

    return float(psi)


# Drift 
def compute_numerical_drift(ref: pd.Series, prod: pd.Series) -> dict:
    """
    Diagnostic signals for numerical features.
    """

    return {
        "reference_mean": ref.mean(),
        "production_mean": prod.mean(),
        "mean_difference": prod.mean() - ref.mean(),
        "reference_std": ref.std(),
        "production_std": prod.std(),
        "reference_missing_rate": ref.isna().mean(),
        "production_missing_rate": prod.isna().mean(),
        "psi": compute_psi(ref, prod),
    }


def compute_categorical_drift(ref: pd.Series, prod: pd.Series) -> dict:
    """
    Diagnostic signals for categorical features.
    """

    ref_dist = ref.value_counts(normalize=True)
    prod_dist = prod.value_counts(normalize=True)

    distribution_shift = {}
    for category in set(ref_dist.index).union(set(prod_dist.index)):
        distribution_shift[str(category)] = {
            "reference_freq": ref_dist.get(category, 0.0),
            "production_freq": prod_dist.get(category, 0.0),
        }

    return {
        "psi": compute_categorical_psi(ref, prod),
        "distribution_shift": distribution_shift,
    }



# Main 
def main():
    for batch_file in sorted(PRODUCTION_DIR.glob("production_batch_*.csv")):
        prod_df = pd.read_csv(batch_file)

        batch_drift = {}

        # Numerical features
        for feature in numerical_features:
            batch_drift[feature] = compute_numerical_drift(
                reference_df[feature], prod_df[feature]
            )

        # Categorical features
        for feature in categorical_features:
            batch_drift[feature] = compute_categorical_drift(
                reference_df[feature], prod_df[feature]
            )

        output_path = OUTPUT_DIR / f"{batch_file.stem}_drift.json"
        pd.Series(batch_drift).to_json(output_path, indent=2)

        print(f"Saved drift report: {output_path}")


if __name__ == "__main__":
    main()