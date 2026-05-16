# Consideraciones Éticas y Limitaciones del Modelo - Telco Customer Churn

Este documento describe los principios éticos, posibles sesgos y limitaciones técnicas asociados con el desarrollo e implementación del modelo de predicción de abandono de clientes (*Churn*).

## 1. Consideraciones Éticas

### A. Privacidad y Confidencialidad de los Datos
* **Anonimización:** El dataset original incluye una columna `customerID`. Como medida de privacidad de los datos (alineada con regulaciones como GDPR o leyes locales de protección de datos), este identificador **ha sido eliminado** en la etapa de ingeniería de datos (`data_loader.py`). El modelo no utiliza datos de identificación personal (PII) para realizar sus predicciones.
* **Uso Interno:** Las predicciones de este modelo deben utilizarse exclusivamente para mejorar la experiencia del cliente y diseñar estrategias de retención legítimas (como ofertas o descuentos), y nunca para penalizar o discriminar activamente a los usuarios.

### B. Sesgo y Equidad (Fairness)
* **Variables Demográficas:** El dataset contiene variables como `gender` (género), `SeniorCitizen` (adulto mayor) y `Partner`/`Dependents` (estado familiar). Existe el riesgo de que el modelo aprenda correlaciones históricas y sesgadas (por ejemplo, asumir que un género o rango de edad específico tiene mayor tendencia a abandonar el servicio debido a deficiencias en el producto).
* **Mitigación:** Se recomienda monitorear que las tasas de falsos positivos y falsos negativos no perjudiquen desproporcionadamente a un grupo demográfico específico, evitando sesgos discriminatorios en las campañas de marketing automatizadas.

### C. Transparencia y Explicabilidad
* **Modelos de Caja Negra:** El algoritmo `RandomForest`, aunque es altamente preciso, puede actuar como una "caja negra" difícil de interpretar para los tomadores de decisiones de negocio. 
* **Responsabilidad:** No se deben tomar decisiones comerciales drásticas basadas únicamente en la predicción del modelo sin antes validar las métricas de importancia de variables (*feature importances*) para entender *por qué* el modelo clasifica a un cliente en riesgo de fuga.

---

## 2. Limitaciones del Dataset y el Modelo

### A. Limitaciones de los Datos
* **Naturaleza Estática:** El dataset es una fotografía estática en el tiempo. No captura dinámicas del mercado en tiempo real, fluctuaciones económicas, cambios en los precios de la competencia o problemas técnicos temporales en la red que podrían alterar drásticamente el comportamiento de *Churn*.
* **Falta de Contexto Cualitativo:** No se incluyen registros de interacciones directas con soporte al cliente, transcripciones de quejas, ni encuestas de satisfacción (NPS), los cuales son predictores críticos para la fuga de clientes.

### B. Limitaciones Técnicas del Modelo
* **Desbalance de Clases:** El conjunto de datos presenta un desbalance (aproximadamente 73% de permanencia vs 27% de abandono). Aunque se mitiga usando técnicas como pesos balanceados (`class_weight='balanced'`), el modelo podría tener una tendencia natural a fallar más en la detección de la clase minoritaria (falsos negativos).
* **Capacidad de Generalización:** Este modelo está sobreajustado a los patrones de esta empresa de telecomunicaciones en particular y no debe ser utilizado para predecir el comportamiento de clientes en otras industrias o mercados geográficos diferentes.