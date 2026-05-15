import yaml
from src.data_loader import load_and_preprocess_data
# Importamos la función unificada de model_trainer.py
from src.model_trainer import train_and_save_model 
from src.experiment_logger import log_experiment 

def main():
    # 1. Cargar configuración
    with open("config/params.yaml", "r") as file:
        config = yaml.safe_load(file)

    print("Configuración cargada correctamente")

    # 2. Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    print("\nShapes:")
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # 3. Entrenar, Evaluar y Guardar
    metrics = train_and_save_model(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        config
    )

    # 4. Guardar resultados
    log_experiment(config, metrics)

    print("\n--- Proceso de ML completado con éxito ---")
    print(f"Resultado final: {metrics}")

if __name__ == "__main__":
    main()

    