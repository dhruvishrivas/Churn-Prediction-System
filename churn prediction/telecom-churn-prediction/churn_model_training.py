
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_data_preprocessing import load_and_preprocess_data




BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "telecom_churn_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
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


def build_model():
    """Build the preprocessing and classification pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model


def train_and_evaluate():
    """Load data, train the model, evaluate it and save the pipeline."""

    print("=" * 60)
    print("TELECOM CUSTOMER CHURN - MODEL TRAINING")
    print("=" * 60)

    # Load karo model ke liye data
    X, y = load_and_preprocess_data()

    expected_features = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Required model features are missing: {missing_features}"
        )

    X = X[expected_features].copy()

    if X.empty:
        raise ValueError("The dataset contains no usable records.")

    if y.nunique() < 2:
        raise ValueError(
            "Training requires both churn classes: 0 and 1."
        )

    #
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Total records: {len(X)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Training churn rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing churn rate: {y_test.mean() * 100:.2f}%")

   
    model = build_model()

    print("\nTraining Logistic Regression model...")
    model.fit(X_train, y_train)

    # prediction karne ke liye test data using
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

   
    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        ),
    }

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: "
              f"{metric_value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Stayed", "Churned"],
            zero_division=0,
        )
    )

    print("Confusion Matrix (actual rows, predicted columns):")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )
    )

    #data cleaning hone ke baad ab load karo 
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully: {MODEL_PATH}")
    print("Currency features: INR")
    print("Conversion used during preprocessing: 1 USD = 97 INR")
    print("Training completed.")

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
