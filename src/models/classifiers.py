from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = SVC(
        kernel="rbf",
        C=1.0,
        probability=True,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
