from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def prepare_data():
    """Charge, sépare et encode les données, puis retourne le préprocesseur ajusté."""
    # Résout le chemin depuis la racine du projet, quel que soit le dossier courant.
    dataset_path = Path(__file__).resolve().parents[2] / "dataset" / "dataset.csv"
    df = pd.read_csv(dataset_path)

    if "Depression_Status" not in df.columns:
        raise ValueError("Dataset must contain a 'Depression_Status' column.")
    if df.empty:
        raise ValueError("Dataset contains no observations.")

    # Sépare les variables explicatives de la cible et convertit les classes en étiquettes.
    X = df.drop("Depression_Status", axis=1)
    y = df["Depression_Status"].map(
        {
            "Stable / Not Depressed": 0,
            "At Risk / Depressed": 1,
        }
    )
    if y.isnull().any():
        raise ValueError("Target contains unknown categories.")

    # Conserve les proportions des classes dans les deux ensembles.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Encode les colonnes catégorielles et ignore les catégories inconnues à l'ajustement.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                X.columns,
            )
        ]
    )

    # Ajuste l'encodeur uniquement sur l'entraînement pour éviter toute fuite de données.
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Retourne le préprocesseur ajusté pour encoder ultérieurement de nouvelles observations.
    return X_train_encoded, X_test_encoded, y_train, y_test, preprocessor
