
import json
from datetime import datetime
from pathlib import Path

import joblib

from src.preprocessing.data_preprocessing import prepare_data

from src.models.classifiers import (
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_svm
)

from src.models.ann import train_ann

from src.models.ensemble import (
    train_random_forest,
    train_gradient_boosting
)

from src.evaluation.metrics import (
    evaluate_model,
    perform_mcnemar_test
)

from src.evaluation.visualization import (
    plot_roc_curves,
    plot_confusion_matrices,
    plot_ann_loss_curve
)

from src.validation.cross_validation import run_cross_validation


PROJECT_PATH = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_PATH / "src" / "results"
CLASS_LABELS = {
    0: "Stable / Not Depressed",
    1: "At Risk / Depressed",
}


def save_models(models, preprocessor):

    saved_models_path = PROJECT_PATH / "src" / "saved_models"
    saved_models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": models["SVM"],
            "preprocessor": preprocessor
        },
        saved_models_path / "svm_model.joblib"
    )

    joblib.dump(
        {
            "model": models["Logistic Regression"],
            "preprocessor": preprocessor
        },
        saved_models_path / "logistic_regression_model.joblib"
    )

    print("\nModels saved successfully.")


def build_evaluation_report(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    models,
    results,
    auc_results,
    cv_results,
    mcnemar_results,
):
    model_reports = {}

    for name, model in models.items():
        metrics = {
            metric_name: float(value)
            for metric_name, value in results[name].items()
            if metric_name != "Confusion Matrix"
        }
        cross_validation = {
            metric_name: {
                "Mean": float(values["Mean"]),
                "Standard Deviation": float(values["Standard Deviation"]),
            }
            for metric_name, values in cv_results[name].items()
        }

        model_report = {
            "parameters": model.get_params(),
            "metrics": metrics,
            "confusion_matrix": results[name]["Confusion Matrix"].tolist(),
            "roc_auc": float(auc_results[name]),
            "cross_validation": cross_validation,
        }
        if name == "ANN" and hasattr(model, "n_iter_"):
            model_report["training_epochs"] = int(model.n_iter_)
        model_reports[name] = model_report

    def class_distribution(labels):
        return {
            label: int((labels == class_index).sum())
            for class_index, label in CLASS_LABELS.items()
        }

    contingency_table = [
        [int(value) for value in row]
        for row in mcnemar_results["Contingency Table"]
    ]

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "total_observations": int(X_train.shape[0] + X_test.shape[0]),
            "training_observations": int(X_train.shape[0]),
            "test_observations": int(X_test.shape[0]),
            "input_features": len(preprocessor.feature_names_in_),
            "encoded_features": int(X_train.shape[1]),
            "class_count": len(set(y_train.tolist()) | set(y_test.tolist())),
            "class_distribution": {
                "training": class_distribution(y_train),
                "test": class_distribution(y_test),
            },
        },
        "models": model_reports,
        "mcnemar": {
            "Contingency Table": contingency_table,
            "Statistic": float(mcnemar_results["Statistic"]),
            "P-value": float(mcnemar_results["P-value"]),
        },
        "graphs": [
            {
                "title": "Courbes ROC",
                "description": "Sensibilité comparée au taux de faux positifs.",
                "file": "roc_curves.png",
                "alt": "Courbes ROC des sept classificateurs.",
            },
            {
                "title": "Matrices de confusion",
                "description": "Prédictions correctes et erreurs par classe.",
                "file": "confusion_matrices.png",
                "alt": "Matrices de confusion des sept classificateurs.",
            },
            {
                "title": "Apprentissage du réseau neuronal",
                "description": "Évolution de la perte pendant l'entraînement de l'ANN.",
                "file": "ann_loss_curve.png",
                "alt": "Courbe de perte d'entraînement de l'ANN.",
            },
        ],
    }


def save_evaluation_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(report_path)


def main():

    print("===== STUDENT MENTAL WELL-BEING ML PROJECT =====")

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("\nDataset preparation successful!")
    print(f"Training observations: {X_train.shape[0]}")
    print(f"Testing observations: {X_test.shape[0]}")
    print(f"Encoded features: {X_train.shape[1]}")

    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    print("\n===== MODEL TRAINING =====")

    models = {}

    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)
    print("Logistic Regression trained successfully.")

    models["Decision Tree"] = train_decision_tree(X_train, y_train)
    print("Decision Tree trained successfully.")

    models["KNN"] = train_knn(X_train, y_train)
    print("KNN trained successfully.")

    models["SVM"] = train_svm(X_train, y_train)
    print("SVM trained successfully.")

    models["ANN"] = train_ann(X_train_dense, y_train)
    print("ANN trained successfully.")

    models["Random Forest"] = train_random_forest(X_train, y_train)
    print("Random Forest trained successfully.")

    models["Gradient Boosting"] = train_gradient_boosting(X_train_dense, y_train)
    print("Gradient Boosting trained successfully.")

    print("\n===== MODEL EVALUATION =====")

    results = {}
    predictions_dict = {}

    for name, model in models.items():

        if name in ["ANN", "Gradient Boosting"]:
            predictions = model.predict(X_test_dense)
        else:
            predictions = model.predict(X_test)

        predictions_dict[name] = predictions

        metrics = evaluate_model(y_test, predictions)

        results[name] = metrics

        print(f"\n===== {name} =====")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
        print(f"Specificity: {metrics['Specificity']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"F1-score: {metrics['F1-score']:.4f}")

        print("Confusion Matrix:")
        print(metrics["Confusion Matrix"])

    print(f"\nANN epochs completed: {models['ANN'].n_iter_}")

    print("\n===== ANN TRAINING LOSS CURVE =====")

    plot_ann_loss_curve(models["ANN"])

    print("\n===== CONFUSION MATRICES =====")

    plot_confusion_matrices(results)

    print("\n===== ROC CURVES =====")

    auc_results = plot_roc_curves(
        models,
        X_test,
        y_test,
        X_test_dense
    )

    print("\n===== McNEMAR TEST =====")

    mcnemar_results = perform_mcnemar_test(
        y_test,
        predictions_dict["SVM"],
        predictions_dict["Logistic Regression"]
    )

    print("\n===== 5-FOLD CROSS-VALIDATION =====")

    cv_results = run_cross_validation()

    print("\n===== CROSS-VALIDATION SUMMARY =====")

    for name, metrics in cv_results.items():

        print(f"\n{name}")

        for metric, values in metrics.items():

            print(
                f"{metric}: {values['Mean']:.4f} "
                f"(+/- {values['Standard Deviation']:.4f})"
            )

    save_models(models, preprocessor)

    report = build_evaluation_report(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        models,
        results,
        auc_results,
        cv_results,
        mcnemar_results,
    )
    report_path = RESULTS_PATH / "evaluation_report.json"
    save_evaluation_report(report, report_path)

    print(f"\nEvaluation report saved to {report_path}.")
    print("\nAll models trained and evaluated successfully.")


if __name__ == "__main__":
    main()
