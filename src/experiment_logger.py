import json
import os
from datetime import datetime

def log_experiment(config, metrics):
    """
    Guarda los resultados del modelo, la configuración y los metadatos
    en un archivo JSON (agregando a la lista si ya existe).
    """
    results_path = config["paths"].get("results_path", "models/experiment_results.json")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Preparar el nuevo registro
    new_record = {
        "timestamp": datetime.now().isoformat(),
        "dataset": config["data"].get("kaggle_dataset", "unknown"),
        "model_name": config["model"]["name"],
        "parameters": {k: v for k, v in config["model"].items() if k != "name"},
        "metrics": metrics
    }
    
    # Leer resultados anteriores si existen
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
        
    # Añadir nuevo registro y guardar
    data.append(new_record)
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"\nResultados del experimento guardados en: {results_path}")
