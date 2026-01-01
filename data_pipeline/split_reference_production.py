"""
Reference / Production Split
- Creating a fixed reference dataset representing deployment-time data
- Generating multiple production batches simulating post-deployment inputs
- Preserve class balance to avoid misleading monitoring signals
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CLEAN_DATA_PATH = Path("data/clean/cleaned_data.csv")
REFERENCE_DATA_PATH = Path("data/reference/reference_data.csv")
PRODUCTION_DIR = Path("data/production_batches")

def main():
    # Loading cleaned data
    df = pd.read_csv(CLEAN_DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' missing from cleaned data")
    
    # Reference vs Production split
    reference_df, production_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df["Churn"]
    )


    # Saving reference data
    REFERENCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(REFERENCE_DATA_PATH, index=False)

    # Creating production batches
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = 500
    for i in range(0, len(production_df), batch_size):
        batch = production_df.iloc[i:i + batch_size]
        batch.to_csv(
            PRODUCTION_DIR / f"production_batch_{i // batch_size}.csv",
            index=False
        )

    print("Reference and production data created.")

if __name__ == "__main__":
    main()

