import json
from pathlib import Path
from threading import Timer
import webbrowser

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


PROJECT_PATH = Path(__file__).resolve().parent
SRC_PATH = PROJECT_PATH / "src"
INTERFACE_PATH = SRC_PATH / "interfaces"
MODELS_PATH = SRC_PATH / "saved_models"
FEATURES_TEMPLATE_PATH = INTERFACE_PATH / "templates"
STATIC_PATH = INTERFACE_PATH / "static"
EVALUATION_REPORT_PATH = SRC_PATH / "results" / "evaluation_report.json"

CHART_METRICS = (
    ("Accuracy", "Doğruluk"),
    ("Sensitivity", "Duyarlılık"),
    ("Specificity", "Belirlilik"),
    ("Precision", "Kesinlik"),
    ("F1-score", "F1 Puanı"),
    ("roc_auc", "AUC ROC"),
)

app = Flask(
    __name__,
    template_folder=str(FEATURES_TEMPLATE_PATH),
    static_folder=str(STATIC_PATH),
)

LABELS = {
    0: "Stabil / Depresif Değil",
    1: "Riskli / Depresif",
}


def load_model_bundle(file_name):
    """Load a trusted local artifact containing a model and its preprocessor."""
    model_path = MODELS_PATH / file_name
    if not model_path.is_file():
        raise FileNotFoundError(f"Kayıtlı model dosyası bulunamadı: {model_path}")

    model_bundle = joblib.load(model_path)
    if not isinstance(model_bundle, dict) or not {
        "model",
        "preprocessor",
    }.issubset(model_bundle):
        raise ValueError(
            f"Kayıtlı model dosyası 'model' ve 'preprocessor' içermelidir: "
            f"{model_path}"
        )

    return model_bundle


svm_data = load_model_bundle("svm_model.joblib")
logistic_data = load_model_bundle("logistic_regression_model.joblib")

svm_model = svm_data["model"]
svm_preprocessor = svm_data["preprocessor"]
logistic_model = logistic_data["model"]
logistic_preprocessor = logistic_data["preprocessor"]

FEATURE_NAMES = tuple(svm_preprocessor.feature_names_in_)
if tuple(logistic_preprocessor.feature_names_in_) != FEATURE_NAMES:
    raise ValueError("The saved models expect different input features.")


def get_feature_categories(preprocessor, feature_names):
    encoder = preprocessor.named_transformers_.get("categorical")
    categories = getattr(encoder, "categories_", None)
    if categories is None or len(categories) != len(feature_names):
        raise ValueError("The saved preprocessor has invalid feature categories.")

    return {
        feature_name: frozenset(str(category) for category in feature_categories)
        for feature_name, feature_categories in zip(feature_names, categories)
    }


FEATURE_CATEGORIES = get_feature_categories(svm_preprocessor, FEATURE_NAMES)
if get_feature_categories(logistic_preprocessor, FEATURE_NAMES) != FEATURE_CATEGORIES:
    raise ValueError("The saved models expect different feature categories.")


def get_predicted_class(probabilities, classes):
    if len(probabilities) == 0 or len(probabilities) != len(classes):
        raise ValueError("Model probabilities do not match its classes.")

    prediction_index = max(
        range(len(probabilities)),
        key=probabilities.__getitem__,
    )
    return int(classes[prediction_index])


@app.route("/", methods=["GET"])
def home():
    with EVALUATION_REPORT_PATH.open(encoding="utf-8") as report_file:
        evaluation_report = json.load(report_file)

    return render_template(
        "index.html",
        evaluation_models=evaluation_report["models"],
        evaluation_generated_at=evaluation_report["generated_at"],
        evaluation_test_count=evaluation_report["dataset"]["test_observations"],
        chart_metrics=CHART_METRICS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 415

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"error": "A JSON object with answers is required."}), 400

    missing_features = [name for name in FEATURE_NAMES if name not in data]
    extra_features = [name for name in data if name not in FEATURE_NAMES]
    if missing_features or extra_features:
        return (
            jsonify(
                {
                    "error": "The submitted answers do not match the model inputs.",
                    "missing_features": missing_features,
                    "unexpected_features": extra_features,
                }
            ),
            400,
        )

    if any(
        not isinstance(data[name], str) or not data[name].strip()
        for name in FEATURE_NAMES
    ):
        return jsonify({"error": "Every answer must be a non-empty string."}), 400

    invalid_features = [
        name for name in FEATURE_NAMES
        if data[name] not in FEATURE_CATEGORIES[name]
    ]
    if invalid_features:
        return (
            jsonify(
                {
                    "error": "Some answers contain unsupported values.",
                    "invalid_features": invalid_features,
                }
            ),
            400,
        )

    features = pd.DataFrame(
        [[data[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    svm_encoded = svm_preprocessor.transform(features)
    logistic_encoded = logistic_preprocessor.transform(features)

    svm_probabilities = svm_model.predict_proba(svm_encoded)[0]
    svm_prediction = get_predicted_class(svm_probabilities, svm_model.classes_)
    svm_percentages = [
        round(float(probability) * 100, 2) for probability in svm_probabilities
    ]

    logistic_probabilities = logistic_model.predict_proba(logistic_encoded)[0]
    logistic_prediction = get_predicted_class(
        logistic_probabilities,
        logistic_model.classes_,
    )
    logistic_percentages = [
        round(float(probability) * 100, 2) for probability in logistic_probabilities
    ]

    return jsonify(
        {
            "SVM": {
                "prediction": LABELS[svm_prediction],
                "probabilities": {
                    LABELS[int(class_label)]: probability
                    for class_label, probability in zip(
                        svm_model.classes_,
                        svm_percentages,
                    )
                },
            },
            "Logistic Regression": {
                "prediction": LABELS[logistic_prediction],
                "probabilities": {
                    LABELS[int(class_label)]: probability
                    for class_label, probability in zip(
                        logistic_model.classes_,
                        logistic_percentages,
                    )
                },
            },
        }
    )


if __name__ == "__main__":
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=False)
