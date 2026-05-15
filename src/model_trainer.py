import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report
)


def get_model(config):
    """
    Fábrica de modelos.
    Permite elegir el modelo desde params.yaml
    """

    model_name = config["model"]["name"]

    if model_name == "RandomForest":

        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["data"]["random_state"]
        )

    elif model_name == "LogisticRegression":

        model = LogisticRegression(
            C=config["model"]["C"],
            max_iter=config["model"]["max_iter"],
            random_state=config["data"]["random_state"]
        )

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model


def train_and_save_model(
    X_train,
    y_train,
    X_test,
    y_test,
    config
):
    """
    Entrena, evalúa y guarda el modelo.
    """

    # Crear modelo
    model = get_model(config)

    # Entrenar
    model.fit(X_train, y_train)

    # Predicciones
    predictions = model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Guardar modelo
    model_path = config["paths"]["model_save"]

    joblib.dump(model, model_path)

    print(f"\nModelo guardado en: {model_path}")

    # Retornar métricas
    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1
    }

    return metrics