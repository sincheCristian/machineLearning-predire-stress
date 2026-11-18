import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app import (
    FEATURE_CATEGORIES,
    FEATURE_NAMES,
    app,
    get_predicted_class,
)
from main import build_evaluation_report, save_evaluation_report


PROJECT_PATH = Path(__file__).resolve().parents[1]


class SavedModelTests(unittest.TestCase):
    def test_saved_models_predict_with_valid_probabilities(self):
        dataset = pd.read_csv(PROJECT_PATH / "dataset" / "dataset.csv")
        features = dataset.drop("Depression_Status", axis=1)
        labels = dataset["Depression_Status"].map(
            {
                "Stable / Not Depressed": 0,
                "At Risk / Depressed": 1,
            }
        )
        _, X_test, _, y_test = train_test_split(
            features,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels,
        )

        model_paths = (
            PROJECT_PATH / "src" / "saved_models" / "svm_model.joblib",
            PROJECT_PATH
            / "src"
            / "saved_models"
            / "logistic_regression_model.joblib",
        )
        for model_path in model_paths:
            with self.subTest(model=model_path.name):
                bundle = joblib.load(model_path)
                model = bundle["model"]
                encoded_test = bundle["preprocessor"].transform(X_test)
                predictions = model.predict(encoded_test)
                probabilities = model.predict_proba(encoded_test)

                self.assertEqual(len(predictions), len(y_test))
                self.assertEqual(
                    probabilities.shape,
                    (len(y_test), len(model.classes_)),
                )
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
                self.assertTrue(set(predictions).issubset({0, 1}))


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.answers = {
            feature: sorted(FEATURE_CATEGORIES[feature])[0]
            for feature in FEATURE_NAMES
        }

    def test_home_displays_all_evaluation_models(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        for model_name in (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "SVM",
            "ANN",
            "Random Forest",
            "Gradient Boosting",
        ):
            self.assertIn(model_name, html)
        self.assertEqual(html.count('aria-label="Matrice de confusion'), 7)

    def test_predict_returns_both_models_and_probabilities(self):
        response = self.client.post("/predict", json=self.answers)

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(set(result), {"SVM", "Logistic Regression"})
        for model_result in result.values():
            self.assertIn(model_result["prediction"], model_result["probabilities"])
            self.assertAlmostEqual(
                sum(model_result["probabilities"].values()),
                100.0,
                places=1,
            )

    def test_predict_rejects_unknown_answer_values(self):
        answers = self.answers.copy()
        answers[FEATURE_NAMES[0]] = "__invalid__"

        response = self.client.post("/predict", json=answers)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["invalid_features"], [FEATURE_NAMES[0]])

    def test_predict_rejects_missing_and_extra_features(self):
        missing_answer = self.answers.copy()
        missing_answer.pop(FEATURE_NAMES[0])
        missing_response = self.client.post("/predict", json=missing_answer)

        extra_answer = self.answers.copy()
        extra_answer["unexpected"] = "value"
        extra_response = self.client.post("/predict", json=extra_answer)

        self.assertEqual(missing_response.status_code, 400)
        self.assertEqual(extra_response.status_code, 400)

    def test_prediction_uses_unrounded_probabilities(self):
        prediction = get_predicted_class([0.49996, 0.50004], [0, 1])

        self.assertEqual(prediction, 1)


class EvaluationReportTests(unittest.TestCase):
    def test_report_contains_json_serializable_evaluation_data(self):
        class DummyPreprocessor:
            feature_names_in_ = ("feature_1", "feature_2")

        class DummyModel:
            def get_params(self):
                return {"random_state": 42}

        results = {
            "Dummy": {
                "Accuracy": 1.0,
                "Sensitivity": 1.0,
                "Specificity": 1.0,
                "Precision": 1.0,
                "F1-score": 1.0,
                "Confusion Matrix": np.array([[1, 0], [0, 1]]),
            }
        }
        report = build_evaluation_report(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            pd.Series([0, 1]),
            pd.Series([0, 1]),
            DummyPreprocessor(),
            {"Dummy": DummyModel()},
            results,
            {"Dummy": 0.9},
            {"Dummy": {"Accuracy": {"Mean": 1.0, "Standard Deviation": 0.0}}},
            {
                "Contingency Table": [[1, 0], [0, 1]],
                "Statistic": 0,
                "P-value": 1.0,
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "evaluation_report.json"
            save_evaluation_report(report, report_path)
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report_path.with_suffix(".tmp").exists())

        self.assertEqual(saved_report["dataset"]["test_observations"], 2)
        self.assertEqual(saved_report["models"]["Dummy"]["confusion_matrix"], [[1, 0], [0, 1]])
