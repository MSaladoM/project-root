import yaml
# Importamos los contratos (funciones) de los Roles 1 y 2
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_save_model

def main():
    # 1. Cargar la configuración desde params.yaml
    try:
        with open('config/params.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: No se encontró el archivo config/params.yaml")
        return

    print("=== Iniciando Pipeline de MLOps ===")
    
    # 2. Ejecutar la parte del Data Engineer (Rol 1)
    print("-> Ejecutando Data Loader...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print(f"   Datos listos. Tamaño del set de entrenamiento: {len(X_train)} filas.")
    
    # 3. Ejecutar la parte del ML Engineer (Rol 2)
    print(f"-> Entrenando modelo: {config['model']['name']}...")
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
    
    # 4. Mostrar resultados finales
    print("=== Pipeline Completado con Éxito ===")
    print(f"   Modelo guardado en: {config['paths']['model_save']}")
    print("   Métricas obtenidas:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - Recall:   {metrics['recall']:.4f}")
    print(f"   - F1 Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()