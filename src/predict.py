
import joblib
import pandas as pd
import yaml
import os
with open("config/params.yaml", "r") as file:
        config = yaml.safe_load(file)




datos_cliente = {
    'customerID':       '1234-ABCD',
    'gender':           'Male',           # 'Male' / 'Female'
    'SeniorCitizen':    0,                # 0 / 1
    'Partner':          'Yes',            # 'Yes' / 'No'
    'Dependents':       'No',             # 'Yes' / 'No'
    'tenure':           12,               # meses con la compañía
    'PhoneService':     'Yes',            # 'Yes' / 'No'
    'MultipleLines':    'No',             # 'No phone service' / 'No' / 'Yes'
    'InternetService':  'DSL',            # 'DSL' / 'Fiber optic' / 'No'
    'OnlineSecurity':   'No',             # 'No internet service' / 'No' / 'Yes'
    'OnlineBackup':     'Yes',            # 'No internet service' / 'No' / 'Yes'
    'DeviceProtection': 'No',             # 'No internet service' / 'No' / 'Yes'
    'TechSupport':      'No',             # 'No internet service' / 'No' / 'Yes'
    'StreamingTV':      'No',             # 'No internet service' / 'No' / 'Yes'
    'StreamingMovies':  'No',             # 'No internet service' / 'No' / 'Yes'
    'Contract':         'Month-to-month', # 'Month-to-month' / 'One year' / 'Two year'
    'PaperlessBilling': 'Yes',            # 'Yes' / 'No'
    'PaymentMethod':    'Electronic check', # 'Electronic check' / 'Mailed check' /
                                            # 'Bank transfer (automatic)' /
                                            # 'Credit card (automatic)'
    'MonthlyCharges':   65.5,
    'TotalCharges':     '786.0',          # puede venir como string
}



# ─────────────────────────────────────────────
#  PREPROCESAMIENTO  (replica load_and_preprocess_data)
# ─────────────────────────────────────────────

def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eliminar customerID
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Limpiar TotalCharges (espacios → NaN → mediana)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', pd.NA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Codificación binaria  ← Female=1, Male=0 (igual que el entrenamiento)
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})

    # One-hot encoding con drop_first=True
    df = pd.get_dummies(df, drop_first=True)

    # Columnas que espera el modelo (en orden)
    columnas_modelo = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    ]

    # Agregar columnas faltantes con 0 (categorías no presentes en el input)
    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = 0

    return df[columnas_modelo]


# ─────────────────────────────────────────────
#  PREDICCIÓN
# ─────────────────────────────────────────────

df_input     = pd.DataFrame([datos_cliente])
X            = preprocesar(df_input)
model_path=config["paths"]["model_save"]
if not os.path.exists(model_path):
      print("Error al cargar el modelo, intenta correr main para generarlo")
else:
    modelo = joblib.load(model_path)
    prediccion   = modelo.predict(X)[0]
    print(f"{'El cliente abandonara el servicio' if prediccion == 1 else 'El cliente continuara usando el servicio'}")

    
