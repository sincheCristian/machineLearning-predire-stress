
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def run_cross_validation():

    dataset_path = Path(__file__).resolve().parents[2] / "dataset" / "dataset.csv"
    df = pd.read_csv(dataset_path)

    X = df.drop("Depression_Status", axis=1)

    y = df["Depression_Status"].map({
        "Stable / Not Depressed": 0,
        "At Risk / Depressed": 1
    })

    if y.isnull().any():
        raise ValueError("Target contains unknown categories.")

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform"
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        ),     
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=200,
            tol=0.0,
            n_iter_no_change=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "Accuracy": "accuracy",
        "Sensitivity": make_scorer(
            recall_score,
            pos_label=1,
            zero_division=0
        ),
        "Specificity": make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0
        ),
        "F1-score": "f1"
    }

    results = {}

    print("\n===== 5-FOLD CROSS-VALIDATION =====")

    for name, model in models.items():

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    X_train.columns
                )
            ]
        )

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1
        )

        results[name] = {}

        print(f"\n===== {name} =====")

        for metric in scoring:

            mean_score = np.mean(scores[f"test_{metric}"])
            std_score = np.std(scores[f"test_{metric}"])

            results[name][metric] = {
                "Mean": mean_score,
                "Standard Deviation": std_score
            }

            print(
                f"{metric}: {mean_score:.4f} "
                f"(+/- {std_score:.4f})"
            )

    return results
