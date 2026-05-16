import yaml
from src.data_loader import load_and_preprocess_data
# Importamos la función unificada de model_trainer.py
from src.model_trainer import train_and_save_model 
from src.experiment_logger import log_experiment 
from src.predict import preprocesar
from pydantic import BaseModel
import pandas as pd
import os
import joblib
from typing import Literal
from fastapi import FastAPI

app = FastAPI()
class ClienteRequest(BaseModel):
    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service","No","Yes"]
    InternetService: Literal["DSL","Fiber optic","No"]
    OnlineSecurity: Literal["No internet service","No","Yes"]
    OnlineBackup: Literal["No internet service","No","Yes"]
    DeviceProtection: Literal["No internet service","No","Yes"]
    TechSupport: Literal["No internet service","No","Yes"]
    StreamingTV: Literal["No internet service","No","Yes"]
    StreamingMovies: Literal["No internet service","No","Yes"]
    Contract: Literal["Month-to-month","One year","Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check","Mailed check",
    "Bank transfer (automatic)","Credit card (automatic)"]
    MonthlyCharges: float
    TotalCharges: float


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
@app.post("/Predecir")
async def generar_respuesta(cliente: ClienteRequest):
    with open("config/params.yaml", "r") as file:
        config = yaml.safe_load(file)
    df = pd.DataFrame([cliente.model_dump()])
    X            = preprocesar(df)
    model_path=config["paths"]["model_save"]
    if not os.path.exists(model_path):
        return("Error al cargar el modelo, intenta correr main para generarlo")
    else:
        modelo = joblib.load(model_path)
        prediccion   = modelo.predict(X)[0]
        return(f"{'El cliente abandonara el servicio' if prediccion == 1 else 'El cliente continuara usando el servicio'}")
@app.post("/Ejecutar")
async def ejecutar():
    main()