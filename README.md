
# 📉 Telco Customer Churn - Pipeline MLOps

Este proyecto implementa un pipeline de Machine Learning Operations (MLOps) de extremo a extremo para predecir el abandono de clientes (Churn) en una empresa de telecomunicaciones. 

El proyecto está diseñado bajo buenas prácticas de ingeniería de software, separando la configuración, la carga de datos, el entrenamiento del modelo y la inferencia en módulos distintos, todo orquestado a través de un archivo YAML centralizado.

## 🗂️ Estructura del Proyecto

El proyecto sigue una arquitectura modular:

* **`config/params.yaml`**: Archivo central que controla las rutas, las divisiones de datos y los hiperparámetros de los modelos sin necesidad de tocar el código de Python.
* **`src/data_loader.py`**: Se encarga de cargar los datos crudos, limpiar valores nulos (mediana en `TotalCharges`), aplicar mapeos binarios y One-Hot Encoding, para finalmente guardar los datos procesados.
* **`src/model_trainer.py`**: "Fábrica de modelos" que lee la configuración, inicializa dinámicamente el algoritmo seleccionado (Random Forest o Regresión Logística), lo entrena y guarda el archivo `.pkl`.
* **`src/main.py`**: Archivo orquestador que ejecuta el pipeline completo paso a paso.
* **`src/predict.py`**: Script de simulación de producción para realizar inferencias sobre nuevos clientes con manejo de errores.
* **`tests/test_pipeline.py`**: Pruebas unitarias para asegurar la integridad de los datos procesados y las métricas devueltas por el modelo.

## 🚀 Cómo ejecutar el proyecto

Asegúrate de estar posicionado en la raíz del proyecto (`D:\project-root\churn_mlops_project>`) y de tener instalado `pandas`, `scikit-learn`, `pyyaml` y `joblib`.

**1. Entrenar el Modelo (Pipeline Completo)**
Para ejecutar la limpieza de datos y el entrenamiento del modelo definido en `params.yaml`:
```bash
python -m src.main

```

*Esto generará los datos limpios en la carpeta `data/processed/` y el modelo entrenado en `models/`.*

**2. Ejecutar Inferencia (Producción)**
Para probar el modelo guardado prediciendo el comportamiento de un cliente de prueba:

```bash
python -m src.predict

```

**3. Correr las Pruebas Unitarias (QA)**
Para validar que el pipeline funciona correctamente a nivel de código:

```bash
python -m unittest tests/test_pipeline.py

```

## 📊 Resultados del Mejor Modelo

Durante las pruebas, se evaluaron dos algoritmos: **Random Forest** y **Logistic Regression**.

El mejor desempeño general para la detección de Churn (priorizando la captura de casos positivos mediante el Recall) lo obtuvo la **Regresión Logística**, superando al Random Forest en ambas métricas principales:

* **Accuracy (Exactitud):** 82.11%
* **Recall (Sensibilidad):** 60.05%
* **F1-Score:** 64.00%

*(Nota: Para futuras iteraciones, se recomienda aplicar un escalado de variables (`StandardScaler`) a las columnas numéricas para optimizar la convergencia del algoritmo).*

---

## 🤖 Contribución de LLM (Inteligencia Artificial)

Este proyecto fue desarrollado y estructurado con la asistencia de un Modelo de Lenguaje Grande (LLM). Las principales contribuciones del LLM a lo largo del ciclo de vida del proyecto incluyeron:

1. **Diseño de Arquitectura MLOps:** Orientación sobre cómo separar correctamente las responsabilidades (Data Loader, Model Trainer, Predictor) evitando scripts monolíticos.
2. **Gestión de Configuración (YAML):** Ayuda para estructurar el archivo `params.yaml` de forma dinámica, permitiendo cambiar de algoritmos y sus hiperparámetros (ej. de `RandomForest` a `LogisticRegression`) sin modificar el código fuente.
3. **Resolución de Errores (Debugging):** Diagnóstico y solución de problemas comunes en Python, como errores de rutas absolutas y relativas (`ModuleNotFoundError`), excepciones de diccionario (`KeyError` en la lectura del YAML), y manejo de errores por hiperparámetros no compatibles en `scikit-learn`.
4. **Desarrollo de Pruebas y Producción (QA):** Redacción de pruebas unitarias estandarizadas usando la librería nativa `unittest` y creación de un script de predicción robusto con manejo adecuado de excepciones (ej. verificar la existencia del archivo `.pkl`).
