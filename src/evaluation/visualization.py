
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curves(models, X_test, y_test, X_test_dense):

    results_path = Path(__file__).resolve().parents[1] / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))

    auc_results = {}

    for name, model in models.items():

        if name in ["ANN", "Gradient Boosting"]:
            X = X_test_dense
        else:
            X = X_test

        probabilities = model.predict_proba(X)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities,
            pos_label=1
        )

        auc_score = roc_auc_score(y_test, probabilities)

        auc_results[name] = auc_score

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Student Mental Well-Being")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

    plt.savefig(results_path / "roc_curves.png", dpi=300)
    plt.show()
    plt.close()

    print("\n===== ROC AUC RESULTS =====")

    for name, score in auc_results.items():
        print(f"{name}: {score:.4f}")

    return auc_results






from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrices(results):

    results_path = Path(__file__).resolve().parents[1] / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for i, name in enumerate(model_names):

        cm = results[name]["Confusion Matrix"]

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Stable", "At Risk"]
        )

        display.plot(
            ax=axes[i],
            colorbar=False,
            values_format="d"
        )

        axes[i].set_title(name, fontsize=12, pad=12)

    for i in range(len(model_names), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Confusion Matrices - Student Mental Well-Being",
        fontsize=17,
        y=0.98
    )

    fig.subplots_adjust(
        top=0.88,
        bottom=0.06,
        hspace=0.75,
        wspace=0.55
    )

    fig.savefig(
        results_path / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)





def plot_ann_loss_curve(model):

    results_path = Path(__file__).resolve().parents[1] / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(model.loss_curve_) + 1),
        model.loss_curve_,
        label="ANN Training Loss"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("ANN Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        results_path / "ann_loss_curve.png",
        dpi=300
    )

    plt.show()
    plt.close()

