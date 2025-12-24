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

def main():
    # Loading trained pipeline
    model = joblib.load(MODEL_PATH)

    # Loading reference data
    df = pd.read_csv(REFERENCE_DATA_PATH)
    X = df.drop(columns=["Churn"])

    # Using a small sample for SHAP
    X_sample = X.sample(n=200, random_state=42)

    # Splitting pipeline into preprocessing + classifier
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    # Transforming data into numeric feature space
    X_transformed = preprocessor.transform(X_sample)

    # Build SHAP explainer on the classifier ONLY
    explainer = shap.Explainer(classifier, X_transformed)

    # Compute SHAP values
    shap_values = explainer(X_transformed)

    # Global Explanation
    shap.summary_plot(
        shap_values.values,
        X_transformed,
        show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_feature_importance.png")
    plt.close()

    # Local Explanation
    shap.plots.waterfall(
        shap_values[0],
        show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_explanation_instance_0.png")
    plt.close()

if __name__ == "__main__":
    main()
