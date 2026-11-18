# Machine Learning-Based Classification of Student Mental Well-Being

Academic project for classifying student mental well-being from an artificial
dataset. It includes preprocessing, model training and evaluation, plus a Flask
interface for comparing the saved SVM and logistic-regression models.

## Project structure

```text
dataset/                 Original dataset (read-only)
src/preprocessing/       Dataset loading, splitting, and encoding
src/models/              Classifier, ensemble, and ANN training
src/evaluation/          Metrics and visualization
src/validation/          Cross-validation
src/saved_models/        Saved SVM and logistic-regression bundles
src/results/             Generated evaluation figures and report
src/interfaces/templates/ Flask page shell and reusable partials
src/interfaces/templates/partials/ Questionnaire and dashboard views
src/interfaces/static/   CSS, JavaScript, and local illustrations
main.py                  Model training and evaluation entry point
app.py                   Flask routes and application launcher
```

## Requirements

- Python 3.9 or newer
- Flask
- joblib
- NumPy
- pandas
- scikit-learn 1.2 or newer, below 2.0
- matplotlib
- statsmodels

Install the Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

From the project root:

```powershell
python main.py
```

The preprocessing function returns the encoded training and test features,
their target labels, and the fitted preprocessor. The encoder is fitted only
on the training split and reused to transform the test split. Five-fold
cross-validation and final test-set evaluation run as part of training.

Training regenerates the saved SVM and logistic-regression models, evaluation
figures, and `src/results/evaluation_report.json`. Re-run training after
changing the dataset, models, or dependencies so the interface displays current
results. Keep the same scikit-learn version for training and serving serialized
models.

After training, run the focused model and API tests from the project root:

```powershell
python -m unittest src.test_saved_models
```

## Web interface

Install the requirements and run training once to create the model artifacts
and evaluation report. Then start Flask from the project root:

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser. Do not open the HTML template
directly as a `file://` page: the `/predict` endpoint is provided by Flask.

The questionnaire displays a three-second analysis loader before showing the
results. Answers are sent to the Flask application for prediction. Completed
reports, including answers and both model results, are saved in that browser's
local storage; they are not synchronized between devices. The project uses an
artificial dataset and its predictions are not a medical diagnosis.
