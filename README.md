# 📡 Proyecto Colaborativo MLOps: Predicción de Churn

Proyecto colaborativo de MLOps enfocado en la predicción de abandono de clientes (Churn) usando Machine Learning y buenas prácticas de ingeniería de software.

---

# 🎯 ¿Qué hace este proyecto?

Este proyecto construye un pipeline de Machine Learning modular para predecir si un cliente de telecomunicaciones abandonará el servicio.

El sistema:

* Carga y limpia datos del dataset Telco Customer Churn
* Preprocesa variables categóricas
* Entrena modelos de Machine Learning
* Evalúa métricas importantes
* Guarda el modelo entrenado
* Permite realizar predicciones posteriores

---

# 📂 Dataset utilizado

Dataset: **Telco Customer Churn**

Problema:

* Clasificación binaria
* Predicción de abandono de clientes (Churn)

Aplicación:

* Retención de clientes
* Marketing personalizado
* Reducción de pérdidas económicas
* Optimización de estrategias comerciales

El archivo CSV debe colocarse en:

```bash
data/raw/
```

Nombre esperado:

```bash
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

# 👥 Integrantes y Roles

| Integrante          | Rol                      |
| ------------------- | ------------------------ |
| Víctor López        | Data Engineer            |
| Christopher Medrano | ML Engineer              |
| James Ubiarco       | MLOps Engineer           |
| Jose Lomeli         | QA & Production Engineer |

---

# 🛠️ Tecnologías utilizadas

* Python
* Pandas
* Scikit-learn
* Joblib
* PyYAML
* Pytest
* KaggleHub

---

# 📂 Estructura del Proyecto

```bash
project-root/
│
├── config/
│   └── params.yaml
│
├── data/
│   └── raw/
│
├── models/
│
├── src/
│   ├── data_loader.py
│   ├── model_trainer.py
│   ├── experiment_logger.py
│   ├── main.py
│   └── predict.py
│
├── tests/
│   └── test_pipeline.py
│
├── DATASET.md
├── ETHICS.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Instalación

## 1. Clonar el repositorio

```bash
git clone https://github.com/Hugolopez5134/project-root.git
```

---

## 2. Entrar a la carpeta

```bash
cd project-root
```

---

## 3. Crear entorno virtual

### Windows

```bash
python -m venv venv
```

---

## 4. Activar entorno virtual

### PowerShell

```bash
.\venv\Scripts\Activate.ps1
```

### CMD

```bash
venv\Scripts\activate
```

---

## 5. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# 🚀 ¿Cómo ejecutar el proyecto?

## Ejecutar el pipeline completo

```bash
python -m src.main
```

Esto realizará:

* Carga de datos
* Limpieza
* Entrenamiento
* Evaluación
* Guardado del modelo

---

# 🤖 Modelos implementados

Actualmente el proyecto soporta:

* RandomForestClassifier
* LogisticRegression

La selección se realiza desde:

```yaml
config/params.yaml
```

Ejemplo:

```yaml
model:
  name: "RandomForest"
```

o

```yaml
model:
  name: "LogisticRegression"
```

---

# 📊 Métricas evaluadas

El pipeline calcula:

* Accuracy
* Recall
* F1-Score

---

# 🧪 Ejecutar pruebas

```bash
pytest
```

---

# 🧠 Uso de LLMs

El equipo utilizó herramientas de IA como apoyo para:

* Depuración de errores
* Organización del pipeline
* Estructuración modular
* Generación de pruebas unitarias
* Mejora de documentación

Herramientas utilizadas:

* ChatGPT

---

# ⚠️ Mayor desafío de integración

El principal reto fue coordinar múltiples ramas y módulos simultáneamente sin sobrescribir cambios entre integrantes.

También fue necesario reorganizar commits y separar correctamente responsabilidades por rama para mantener una estructura profesional de MLOps.

---

# 📌 Estado actual del proyecto

✅ Pipeline funcional
✅ Configuración mediante YAML
✅ Entrenamiento modular
✅ Soporte para múltiples modelos
✅ Logging de experimentos
✅ Tests básicos
✅ Uso de Git con ramas especializadas

---

# 📈 Resultados

Las métricas finales dependerán del modelo seleccionado y la configuración utilizada en `params.yaml`.

---

# 🧾 Archivos importantes

| Archivo                | Función                     |
| ---------------------- | --------------------------- |
| `data_loader.py`       | Limpieza y preprocesamiento |
| `model_trainer.py`     | Entrenamiento y métricas    |
| `main.py`              | Orquestación del pipeline   |
| `predict.py`           | Predicciones futuras        |
| `experiment_logger.py` | Registro de experimentos    |
| `params.yaml`          | Configuración central       |

---

# 🔥 Comando principal del proyecto

```bash
python -m src.main
```
