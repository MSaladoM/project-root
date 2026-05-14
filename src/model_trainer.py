import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(X_train, y_train, config):
    """
    Entrena el modelo RandomForest.
    """

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["data"]["random_state"]
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


def save_model(model, path):
    """
    Guarda el modelo entrenado.
    """

    joblib.dump(model, path)

    print(f"\nModelo guardado en: {path}")

    