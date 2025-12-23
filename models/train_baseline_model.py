import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

REFERENCE_DATA_PATH = Path("data/reference/reference_data.csv")
MODEL_PATH = Path("models/baseline_model.joblib")
METRICS_PATH = Path("models/baseline_metrics.txt")

def main():
    df = pd.read_csv(REFERENCE_DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="most_frequent")),
                 ("encoder", OneHotEncoder(handle_unknown="ignore"))
             ]),
             categorical_features),
            ("num",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="median"))
             ]),
             numerical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_pred)
    roc_auc = roc_auc_score((y_val == "Yes").astype(int), y_pred_proba)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}\n")

    print("Baseline model trained and saved.")

if __name__ == "__main__":
    main()
