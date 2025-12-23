import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/bank_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/bank_data.csv")

def main():
    df = pd.read_csv(RAW_DATA_PATH)

    # Basic preprocessing
    df = df.dropna()

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

if __name__ == "__main__":
    main()
