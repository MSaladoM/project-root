import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config):
    df = pd.read_csv(config["data"]["raw_path"])

    # TotalCharges tiene espacios en blanco en vez de nulos — convertir a numérico y rellenar con la mediana
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df.drop(columns=["customerID"], inplace=True)

    binary_cols = {"gender": {"Male": 1, "Female": 0},
                   "Partner": {"Yes": 1, "No": 0},
                   "Churn": {"Yes": 1, "No": 0}}
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
