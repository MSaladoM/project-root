import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def _get_model(config):
    # 1. Leemos qué modelo está seleccionado en el YAML
    algo = config["model"]["name"]
    
    # 2. Extraemos los parámetros correspondientes
    if algo == "RandomForest":
        p = config["model"].get("params_random_forest", {})
        return RandomForestClassifier(**p)
        
    elif algo == "LogisticRegression":
        p = config["model"].get("params_logistic_regression", {})
        return LogisticRegression(**p)
        
    raise ValueError(f"Algoritmo no soportado: {algo}")

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    # 3. Instanciar el modelo con la función auxiliar
    model = _get_model(config)
    
    # 4. Entrenar
    model.fit(X_train, y_train)

    # 5. Predecir y evaluar
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    # 6. Guardar el modelo en la ruta especificada en el YAML
    joblib.dump(model, config["paths"]["model_save"])
    
    return metrics