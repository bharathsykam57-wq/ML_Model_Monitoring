import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/telco_customer_churn.csv")
CLEAN_DATA_PATH = Path("data/clean/cleaned_data.csv")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges type issue identified in EDA
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop identifier column
    df = df.drop(columns=["customerID"])

    return df

def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df_clean = clean_data(df)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)

    print("Cleaned data saved to:", CLEAN_DATA_PATH)

if __name__ == "__main__":
    main()
