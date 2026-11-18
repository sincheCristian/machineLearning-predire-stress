
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)


def evaluate_model(y_test, predictions):

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    tn, fp, _, _ = cm.ravel()

    accuracy = accuracy_score(y_test, predictions)

    sensitivity = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1,
        "Confusion Matrix": cm
    }


from statsmodels.stats.contingency_tables import mcnemar


def perform_mcnemar_test(y_true, predictions_model_1, predictions_model_2):

    correct_1 = predictions_model_1 == y_true
    correct_2 = predictions_model_2 == y_true

    both_correct = sum(correct_1 & correct_2)
    model_1_only_correct = sum(correct_1 & ~correct_2)
    model_2_only_correct = sum(~correct_1 & correct_2)
    both_wrong = sum(~correct_1 & ~correct_2)

    contingency_table = [
        [both_correct, model_1_only_correct],
        [model_2_only_correct, both_wrong]
    ]

    test_result = mcnemar(
        contingency_table,
        exact=True
    )

    print("\n===== McNEMAR TEST =====")
    print("Contingency Table:")
    print(contingency_table)
    print(f"Statistic: {test_result.statistic}")
    print(f"P-value: {test_result.pvalue:.4f}")

    if test_result.pvalue < 0.05:
        print("Result: Significant difference between the two models.")
    else:
        print("Result: No statistically significant difference between the two models.")

    return {
        "Contingency Table": contingency_table,
        "Statistic": test_result.statistic,
        "P-value": test_result.pvalue
    }

