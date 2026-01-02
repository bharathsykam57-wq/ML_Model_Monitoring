"""
Model explainability script using SHAP.
- Provide global and local explanations for the baseline model
- Support trust, debugging, and monitoring decisions
- Ensure explanations are consistent with training schema
"""

import pandas as pd
import shap
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = Path("models/baseline_model.joblib")
REFERENCE_DATA_PATH = Path("data/reference/reference_data.csv")
OUTPUT_DIR = Path("models/explainability")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
SAMPLE_SIZE = 200


def main():
    # Loading trained pipeline
    model = joblib.load(MODEL_PATH)

    # Loading reference data
    df = pd.read_csv(REFERENCE_DATA_PATH)

    # Schema consistency (must be same training)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    X = df.drop(columns=[TARGET_COLUMN])

    # Sample data for explainability
    X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)


    # Extracting pipeline components
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    # Transform features
    X_transformed = preprocessor.transform(X_sample)

    # Recovering feature names after preprocessing
    cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out()
    )

    num_features = preprocessor.transformers_[1][2]

    feature_names = np.concatenate([cat_features, num_features])

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # SHAP Explainer setup 
    explainer = shap.LinearExplainer(
        classifier,
        X_transformed_df,
        feature_perturbation="interventional"
    )

    shap_values = explainer(X_transformed_df)

    # Global explanation
    shap.summary_plot(
        shap_values.values,
        X_transformed_df,
        show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_feature_importance.png")
    plt.close()

    # Local explanation (single instance)
    shap.plots.waterfall(
    shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],  # Use the first base value
        data=X_transformed_df.iloc[0],
        feature_names=X_transformed_df.columns,
    ),
    show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_explanation_instance_0.png")
    plt.close()

    print("SHAP explainability artifacts generated successfully.")


if __name__ == "__main__":
    main()