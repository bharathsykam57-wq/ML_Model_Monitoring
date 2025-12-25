import pandas as pd
from pathlib import Path

REFERENCE_PATH = Path("data/reference/reference_data.csv")
PRODUCTION_DIR = Path("data/production_batches")
OUTPUT_DIR = Path("monitoring/drift_reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Loading reference data
reference_df = pd.read_csv(REFERENCE_PATH)

categorical_features = reference_df.select_dtypes(include=["object"]).columns
numerical_features = reference_df.select_dtypes(exclude=["object"]).columns

# numerical drift detection
def compute_numerical_drift(ref: pd.Series, prod: pd.Series) -> dict:
    return {
        "reference_mean": ref.mean(),
        "production_mean": prod.mean(),
        "mean_difference": prod.mean() - ref.mean(),
        "reference_std": ref.std(),
        "production_std": prod.std(),
    }

# categorical drift detection
def compute_categorical_drift(ref: pd.Series, prod: pd.Series) -> dict:
    ref_dist = ref.value_counts(normalize=True)
    prod_dist = prod.value_counts(normalize=True)

    drift = {}
    all_categories = set(ref_dist.index).union(set(prod_dist.index))

    for category in all_categories:
        drift[str(category)] = {
            "reference_freq": ref_dist.get(category, 0.0),
            "production_freq": prod_dist.get(category, 0.0),
            "freq_difference": prod_dist.get(category, 0.0)
            - ref_dist.get(category, 0.0),
        }

    return drift

# iterate over production batches
def main():
    for batch_file in sorted(PRODUCTION_DIR.glob("production_batch_*.csv")):
        prod_df = pd.read_csv(batch_file)

        batch_drift = {}
        # compute numerical drift
        for feature in numerical_features:
            batch_drift[feature] = compute_numerical_drift(
                reference_df[feature], prod_df[feature]
            )
        # compute categorical drift
        for feature in categorical_features:
            batch_drift[feature] = compute_categorical_drift(
                reference_df[feature], prod_df[feature]
            )
        # saving drift report
        output_path = OUTPUT_DIR / f"{batch_file.stem}_drift.json"
        pd.Series(batch_drift).to_json(output_path, indent=2)

        print(f"Saved drift report: {output_path}")


if __name__ == "__main__":
    main()
