
from sklearn.neural_network import MLPClassifier


def train_ann(X_train, y_train):

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        tol=0.0,
        n_iter_no_change=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
