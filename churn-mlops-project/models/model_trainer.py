import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def _get_model(config):
    algo = config["model"]["algorithm"]
    if algo == "random_forest":
        p = config["random_forest"]
        return RandomForestClassifier(**p, random_state=42)
    elif algo == "logistic_regression":
        p = config["logistic_regression"]
        return LogisticRegression(**p, random_state=42)
    raise ValueError(f"Algoritmo no soportado: {algo}")

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    model = _get_model(config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    joblib.dump(model, config["model"]["output_path"])
    return metrics