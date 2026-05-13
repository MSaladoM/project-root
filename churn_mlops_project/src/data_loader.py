import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(config):
    # 1. Cargar datos
    df = pd.read_csv(config["data"]["raw_path"])

    # 2. Limpieza y Transformación (Instrucciones)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df.drop(columns=["customerID"], inplace=True)

    binary_cols = {"gender": {"Male": 1, "Female": 0},
                   "Partner": {"Yes": 1, "No": 0},
                   "Churn": {"Yes": 1, "No": 0}}
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    # 3. Convertir el RESTO de las columnas de texto a números (One-Hot Encoding)
    # Esto evita que scikit-learn arroje un error en el Rol 2
    df = pd.get_dummies(df, drop_first=True)

    # 4. Separar X e y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 5. División (Asegúrate de usar "data" y no "split" si así está en tu YAML)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # 6. GUARDAR LOS ARCHIVOS PROCESADOS (NUEVO)
    # Crea la carpeta si no existe para evitar errores
    os.makedirs("data/processed", exist_ok=True) 
    
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("   -> Datos procesados guardados en data/processed/")

    return X_train, X_test, y_train, y_test