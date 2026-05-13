import joblib
import os
import pandas as pd

def predict_new_client():
    model_path = "models/model.pkl"
    
    # 1. Manejo de errores: Verificar si el modelo existe
    if not os.path.exists(model_path):
        print(f"ERROR CRÍTICO: No se encontró el modelo en '{model_path}'.")
        print("Por favor, ejecuta primero 'python -m src.main' para entrenar el modelo.")
        return

    # 2. Cargar el modelo
    try:
        model = joblib.load(model_path)
        print(f"-> Modelo cargado exitosamente desde '{model_path}'.")
    except Exception as e:
        print(f"ERROR: Fallo al cargar el modelo. Detalles: {e}")
        return

    # 3. Cliente de ejemplo (Usamos uno real de nuestro set de pruebas procesado)
    try:
        X_test = pd.read_csv("data/processed/X_test.csv")
        # Seleccionamos la primera fila como nuestro cliente de ejemplo (como un DataFrame de 1 fila)
        nuevo_cliente = X_test.iloc[[0]] 
    except FileNotFoundError:
        print("ERROR: No se encontraron los datos de prueba en 'data/processed/X_test.csv'.")
        return

    # 4. Realizar la predicción
    prediccion = model.predict(nuevo_cliente)
    
    # Extra: Obtener la probabilidad para dar un resultado más profesional
    probabilidad = model.predict_proba(nuevo_cliente)[0][1]

    # 5. Mostrar resultados
    print("\n=== Resultado de la Predicción ===")
    if prediccion[0] == 1:
        print(f" ALERTA: El cliente PROBABLEMENTE CANCELARÁ el servicio (Churn = Yes).")
        print(f" Probabilidad de abandono: {probabilidad:.1%}")
    else:
        print(f" SEGURO: El cliente PROBABLEMENTE SE QUEDARÁ (Churn = No).")
        print(f" Probabilidad de abandono: {probabilidad:.1%}")

if __name__ == "__main__":
    predict_new_client()