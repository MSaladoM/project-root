import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def _get_model(config):
    # 1. Leemos el nombre del modelo tal como está en el YAML
    algo = config["model"]["name"]
    
    # 2. Leemos los hiperparámetros desde la sección 'params'
    p = config["model"].get("params", {})
    
    # 3. Comparamos con los strings exactos
    if algo == "RandomForest":
        # Nota: si random_state ya viene en el YAML (dentro de p), no hace falta forzarlo aquí
        return RandomForestClassifier(**p)
    elif algo == "LogisticRegression":
        return LogisticRegression(**p)
    raise ValueError(f"Algoritmo no soportado: {algo}")

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    # Instanciar y entrenar
    model = _get_model(config)
    model.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # 4. Guardar usando la ruta correcta de paths
    joblib.dump(model, config["paths"]["model_save"])
    
    return metrics