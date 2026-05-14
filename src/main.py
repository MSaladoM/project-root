import yaml

from src.data_loader import load_and_preprocess_data

from src.model_trainer import (
    train_model,
    evaluate_model,
    save_model
)


def main():

    # Cargar configuración
    with open("config/params.yaml", "r") as file:
        config = yaml.safe_load(file)

    print("Configuración cargada correctamente")

    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    print("\nShapes:")
    print(X_train.shape)
    print(X_test.shape)

    # Entrenar modelo
    model = train_model(X_train, y_train, config)

    # Evaluar modelo
    evaluate_model(model, X_test, y_test)

    # Guardar modelo
    save_model(
        model,
        config["paths"]["model_save"]
    )


if __name__ == "__main__":
    main()