
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from churn_data_preprocessing import load_and_preprocess_data


BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
MODEL_PATH = BASE_DIR / "model" / "telecom_churn_model.pkl"

app = Flask(
    __name__,
    template_folder=str(UI_DIR),
    static_folder=str(UI_DIR),
    static_url_path="/static",
)

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

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RISK_THRESHOLD = 0.70




def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run: python churn_model_training.py"
        )

    return joblib.load(MODEL_PATH)


model = load_model()


 # kis sey kya map honga

INTERNET_MAPPING = {
    "Recharge": "DSL",
    "Broadband": "Fiber optic",
    "DSL": "DSL",
    "Fiber optic": "Fiber optic",
    "No Internet": "No",
    "No": "No",
}

CONTRACT_MAPPING = {
    "Monthly Plan": "Month-to-month",
    "1-Year Plan": "One year",
    "2-Year Plan": "Two year",
    "Month-to-month": "Month-to-month",
    "One year": "One year",
    "Two year": "Two year",
}


PAYMENT_MAPPING = {
    "UPI": "Electronic check",
    "UPI / Electronic Payment": "Electronic check",
    "Electronic Payment": "Electronic check",
    "Electronic check": "Electronic check",
    "Bank Transfer": "Bank transfer (automatic)",
    "Bank transfer": "Bank transfer (automatic)",
    "Bank transfer (automatic)": "Bank transfer (automatic)",
    "Credit Card": "Credit card (automatic)",
    "Credit card": "Credit card (automatic)",
    "Credit card (automatic)": "Credit card (automatic)",
    "Mailed check": "Mailed check",
}



def validate_customer(customer):
    if not isinstance(customer, dict):
        raise ValueError("Request body must contain a JSON object.")

#yeh field ui me nhi 
    defaults = {
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PaperlessBilling": "Yes",
        "PhoneService": "Yes",
    }

    customer_data = {**defaults, **customer}

    required_fields = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in customer
    ]

    if missing_fields:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing_fields)
        )
    
    customer_data["InternetService"] = INTERNET_MAPPING.get(
        str(customer_data["InternetService"]).strip(),
        customer_data["InternetService"],
    )

    customer_data["Contract"] = CONTRACT_MAPPING.get(
        str(customer_data["Contract"]).strip(),
        customer_data["Contract"],
    )

    customer_data["PaymentMethod"] = PAYMENT_MAPPING.get(
        str(customer_data["PaymentMethod"]).strip(),
        customer_data["PaymentMethod"],
    )

    row = {}

    for field in NUMERIC_FEATURES:
        try:
            value = float(customer_data[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a valid number.")

        if pd.isna(value) or value < 0:
            raise ValueError(
                f"{field} must be a non-negative number."
            )

        row[field] = value

    if row["tenure"] > 72:
        raise ValueError("Tenure must be between 0 and 72 months.")


    allowed_values = {
        "gender": {"Female", "Male"},
        "SeniorCitizen": {"No", "Yes"},
        "Partner": {"No", "Yes"},
        "Dependents": {"No", "Yes"},
        "PhoneService": {"No", "Yes"},
        "MultipleLines": {
            "No", "Yes", "No phone service"
        },
        "InternetService": {
            "DSL", "Fiber optic", "No"
        },
        "OnlineSecurity": {
            "No", "Yes", "No internet service"
        },
        "OnlineBackup": {
            "No", "Yes", "No internet service"
        },
        "DeviceProtection": {
            "No", "Yes", "No internet service"
        },
        "TechSupport": {
            "No", "Yes", "No internet service"
        },
        "StreamingTV": {
            "No", "Yes", "No internet service"
        },
        "StreamingMovies": {
            "No", "Yes", "No internet service"
        },
        "Contract": {
            "Month-to-month", "One year", "Two year"
        },
        "PaperlessBilling": {"No", "Yes"},
        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
    }

    for field, choices in allowed_values.items():
        value = str(customer_data[field]).strip()

        if value not in choices:
            raise ValueError(
                f"Invalid value for {field}: {value}"
            )

        row[field] = value

    

    return pd.DataFrame([row], columns=ALL_FEATURES)


# last me recommendation 

def get_recommendation(risk_level):
    if risk_level == "High":
        return (
            "Contact the customer, understand their problems, "
            "and offer help to keep them."
        )

    if risk_level == "Medium":
        return (
            "Check if the customer has any problems and "
            "offer support when needed."
        )

    return (
        "Keep the customer happy with good service "
        "and regular support."
    )



@app.route("/")
def home():
    return render_template("churn_dashboard.html")


# dataset model se aayi hui values

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    try:
        X, y = load_and_preprocess_data()

        total_customers = int(len(y))
        churned_count = int(y.sum())
        stayed_count = total_customers - churned_count
        churn_rate = float(y.mean()) if total_customers else 0.0

        contract_rates = y.groupby(X["Contract"]).mean()

        contract_churn = {
            str(contract): float(rate)
            for contract, rate in contract_rates.items()
        }

        probabilities = model.predict_proba(X)[:, 1]
        high_risk_customers = int(
            (probabilities >= RISK_THRESHOLD).sum()
        )

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
        }

        return jsonify({
            "total_customers": total_customers,
            "churned_count": churned_count,
            "stayed_count": stayed_count,
            "churn_rate": churn_rate,
            "high_risk_customers": high_risk_customers,
            "recall": metrics["recall"],
            "contract_churn": contract_churn,
            "metrics": metrics,
        })

    except Exception:
        app.logger.exception("Dashboard loading failed")
        return jsonify({
            "error": (
                "Unable to load dashboard data. "
                "Check the server terminal."
            )
        }), 500


# customer button dbaye toh preict api ko lker aao aur response yha se dege

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        customer = request.get_json(silent=True)

        if customer is None:
            return jsonify({
                "error": "Please submit valid JSON customer data."
            }), 400

        customer_df = validate_customer(customer)

        prediction = int(model.predict(customer_df)[0])
        probabilities = model.predict_proba(customer_df)[0]
        churn_probability = float(probabilities[1])

        if churn_probability >= RISK_THRESHOLD:
            risk_level = "High"
        elif churn_probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "prediction": prediction,
            "churn_probability": churn_probability,
            "risk_level": risk_level,
            "recommendation": get_recommendation(risk_level),
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Customer prediction failed")
        return jsonify({
            "error": (
                "Prediction failed. Check the server terminal."
            )
        }), 500



if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
