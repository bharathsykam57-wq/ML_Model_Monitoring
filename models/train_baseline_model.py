import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer

# Paths
REFERENCE_DATA_PATH = Path("data/reference/reference_data.csv")
MODEL_OUTPUT_PATH = Path("models/baseline_model.joblib")
METRICS_OUTPUT_PATH = Path("models/baseline_metrics.txt")

# Loading reference data
df = pd.read_csv(REFERENCE_DATA_PATH)

# Schema consistency fixes
# Converting TotalCharges to numeric (known issue from EDA)
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Ensuring categorical columns are strings
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str)

# target split
TARGET_COLUMN = "Churn"
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            numerical_features,
        ),
    ]
)

# Model pipeline

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train / validation split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model

model.fit(X_train, y_train)

# Evaluate baseline
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

report = classification_report(y_val, y_pred)
roc_auc = roc_auc_score((y_val == "Yes").astype(int), y_pred_proba)

# Saving artifacts

MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)

with open(METRICS_OUTPUT_PATH, "w") as f:
    f.write("Baseline Model Performance\n\n")
    f.write(report)
    f.write(f"\nROC AUC: {roc_auc:.4f}\n")

print("Baseline model training complete.")
