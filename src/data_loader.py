import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config):

    # Ruta del dataset
    data_path = config["paths"]["data_path"]

    # Cargar dataset
    df = pd.read_csv(data_path)

    print("Dataset cargado correctamente")
    print(df.head())

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

    return X_train, X_test, y_train, y_test

