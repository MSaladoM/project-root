import os
import shutil
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config):

    # Ruta del dataset
    data_path = config["paths"]["data_path"]

    if not os.path.exists(data_path):
        print(f"Dataset no encontrado en {data_path}. Descargando de Kaggle...")
        download_path = kagglehub.dataset_download(config["data"]["kaggle_dataset"])
        print("Path to downloaded dataset files:", download_path)
        
        # Crear la carpeta si no existe
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Copiar el archivo CSV a la ruta definida en config
        for file in os.listdir(download_path):
            if file.endswith(".csv"):
                shutil.copy(os.path.join(download_path, file), data_path)
                break

    # Cargar dataset
    df = pd.read_csv(data_path)

    print("Dataset cargado correctamente")
    print(df.head())
    print(df.info())
    # -----------------------------
    # Limpieza de TotalCharges
    # -----------------------------

    # Reemplazar espacios vacíos por NaN
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)

    # Convertir a numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    # Rellenar NaN con mediana
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # -----------------------------
    # Eliminar columnas innecesarias
    # -----------------------------

    df = df.drop(columns=["customerID"])

    # -----------------------------
    # Codificación binaria
    # -----------------------------

    binary_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]

    for col in binary_columns:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0,
            "Female": 1,
            "Male": 0
        })

    # -----------------------------
    # One-hot encoding
    # -----------------------------

    df = pd.get_dummies(df, drop_first=True)

    # -----------------------------
    # Separar variables
    # -----------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------
    # Train/Test split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    print("Preprocesamiento completado")
    print(df.info())
    return X_train, X_test, y_train, y_test

