import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    # 1. Leer el interruptor desde tu params.yaml
    model_name = config["model"]["name"]

    # 2. Elegir el modelo basado en la configuración
    if model_name == "RandomForest":
        print("Inicializando Random Forest...")
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["split"]["random_state"]
        )
    
    elif model_name == "LogisticRegression":
        print("Inicializando Regresión Logística...")
        model = LogisticRegression(
            max_iter=1000, 
            random_state=config["split"]["random_state"]
        )
    
    else:
        # Buena práctica: lanzar un error si escribes mal el nombre en el YAML
        raise ValueError(f"El modelo {model_name} no está soportado.")

    # 3. Entrenar el modelo elegido
    model.fit(X_train, y_train)

    # 4. Calcular métricas
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # 5. Guardar el modelo
    os.makedirs(os.path.dirname(config["paths"]["model_save"]), exist_ok=True)
    joblib.dump(model, config["paths"]["model_save"])

    return metrics