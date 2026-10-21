
from pathlib import Path

import pandas as pd



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_CANDIDATES = [
    DATA_DIR / "telco_customer_churn.csv",
    DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

USD_TO_INR = 97.0

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def find_dataset() -> Path:
    """Locate the original IBM Telco Customer Churn CSV."""

    for path in DATASET_CANDIDATES:
        if path.is_file():
            return path

    expected_files = "\n".join(
        f"- {path}" for path in DATASET_CANDIDATES
    )

    raise FileNotFoundError(
        "Telco Customer Churn dataset was not found.\n"
        "Place the CSV file inside the data/ folder.\n"
        f"Accepted filenames:\n{expected_files}"
    )


def load_and_preprocess_data(csv_path=None):
    """
    Load and clean the dataset.

    Returns:
        X: Features as a pandas DataFrame.
        y: Binary churn target as a pandas Series.

    Currency conversion:
        MonthlyCharges and TotalCharges are converted from USD
        to INR using the fixed project rate of 1 USD = 97 INR.

    The conversion is applied only to the in-memory data.
    The original CSV remains unchanged.
    """

    dataset_path = Path(csv_path) if csv_path else find_dataset()

    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    # colum cleaning
    df.columns = df.columns.str.strip()

    required_columns = (
        [TARGET_COLUMN, ID_COLUMN]
        + NUMERIC_COLUMNS
        + CATEGORICAL_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset does not match the expected IBM Telco "
            f"schema. Missing columns: {missing_columns}"
        )

    # duplicate records.
    df = df.drop_duplicates().copy()

    # same id waale record ko hta.
    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first").copy()

    # jo  valid churn label nhi hia unko htao
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype("string")
        .str.strip()
    )

    df = df[
        df[TARGET_COLUMN].isin(["Yes", "No"])
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid customer records remain after validating "
            "the Churn column."
        )

    
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Missing TotalCharges waalo ko unknown , negative waalo ko reject
    
    for column in NUMERIC_COLUMNS:
        df.loc[df[column] < 0, column] = float("nan")

    # usd to inr
    for column in ["MonthlyCharges", "TotalCharges"]:
        df[column] = (
            df[column] * USD_TO_INR
        ).round(2)

 
    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    df["SeniorCitizen"] = df["SeniorCitizen"].map(
        {
            0: "No",
            1: "Yes",
            "0": "No",
            "1": "Yes",
        }
    ).astype("string")

   
    X = df.drop(
        columns=[TARGET_COLUMN, ID_COLUMN]
    ).copy()

    y = df[TARGET_COLUMN].map(
        {
            "No": 0,
            "Yes": 1,
        }
    ).astype("int8")

    y.name = TARGET_COLUMN

    if len(X) != len(y):
        raise RuntimeError(
            "Feature and target row counts do not match."
        )

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess_data()

    print("Data preprocessing completed successfully.")
    print(f"Total customers: {len(X)}")
    print(f"Total features: {len(X.columns)}")
    print(f"Churned customers: {int(y.sum())}")
    print(f"Customers retained: {int((y == 0).sum())}")
    print(
        "Churn rate: "
        f"{y.mean() * 100:.2f}%"
    )
    print(
        "MonthlyCharges currency: INR "
        f"(conversion rate: 1 USD = {USD_TO_INR} INR)"
    )
    print("\nFeature columns:")
    print(list(X.columns))
    print("\nMissing values by column:")
    print(X.isnull().sum())
