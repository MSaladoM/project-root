# Documentación del Código Fuente (src/)

Por ahora, este documento detalla el funcionamiento de la fábrica de modelos.

## Model Trainer (`model_trainer.py`)

### ¿Cómo funciona la fábrica de modelos?

La fábrica de modelos está implementada en la función `get_model(config)` del archivo `model_trainer.py`. Funciona mediante el patrón de diseño *Factory* (Fábrica) de una forma muy sencilla:

1. **Lectura de configuración:** Primero, lee el nombre del modelo que quieres usar desde el diccionario de configuración (`model_name = config["model"]["name"]`), el cual proviene directamente del archivo `config/params.yaml`.
2. **Estructura de control (if/elif):** Utiliza una serie de condicionales para verificar qué modelo se ha solicitado.
3. **Instanciación:** 
   - Si el nombre es `"RandomForest"`, crea un objeto `RandomForestClassifier` de Scikit-Learn y le inyecta los hiperparámetros (como `n_estimators` y `max_depth`) que también saca del diccionario de configuración.
   - Si el nombre es `"LogisticRegression"`, crea un objeto `LogisticRegression` pasándole sus hiperparámetros correspondientes (`C` y `max_iter`).
4. **Manejo de errores:** Si el nombre indicado en el YAML no coincide con ninguno de los definidos, lanza un error (`ValueError`) avisando que el modelo no está soportado.
5. **Retorno:** Finalmente, devuelve el objeto del modelo ya configurado y listo para ser entrenado con el método `.fit()`.

---

### ¿Qué tan fácil es añadir nuevos modelos?

Es **extremadamente fácil**. El código está diseñado de forma modular para que puedas escalar la cantidad de algoritmos disponibles sin tocar el resto del pipeline (ni el preprocesamiento, ni el entrenamiento, ni el guardado). 

Para añadir un nuevo modelo (por ejemplo, un *Support Vector Classifier* de Scikit-Learn), solo tienes que seguir **3 pasos**:

1. **Importar el algoritmo:** En la parte superior de `model_trainer.py`, agregas la importación:
   ```python
   from sklearn.svm import SVC
   ```

2. **Añadir el condicional en la fábrica:** Dentro de la función `get_model`, agregas un nuevo `elif` justo antes del `else`:
   ```python
   elif model_name == "SVC":
       model = SVC(
           kernel=config["model"].get("kernel", "rbf"), # Puedes usar .get() por seguridad
           C=config["model"]["C"],
           random_state=config["data"]["random_state"]
       )
   ```

3. **Modificar el YAML:** Por último, en el archivo `config/params.yaml`, simplemente cambias el nombre y ajustas los hiperparámetros:
   ```yaml
   model:
     name: "SVC"
     C: 1.0
     kernel: "linear"
   ```

Con solo hacer eso, el flujo principal funcionará automáticamente con el nuevo modelo.
