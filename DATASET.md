# DATASET.md

# Telco Customer Churn Dataset

## Descripción del Dataset

El proyecto utiliza el dataset **Telco Customer Churn**, disponible en Kaggle.

Este dataset contiene información de clientes de una empresa de telecomunicaciones y tiene como objetivo predecir si un cliente abandonará el servicio (Churn).

### Características principales

- Total de filas: 7,043 clientes
- Total de columnas: 21 variables
- Tipo de problema: Clasificación binaria

### Variable objetivo

- `Churn`
  - Yes = el cliente abandonó el servicio
  - No = el cliente continúa usando el servicio

### Variables incluidas

El dataset contiene variables relacionadas con:

- Información demográfica
  - gender
  - SeniorCitizen
  - Partner
  - Dependents

- Servicios contratados
  - PhoneService
  - InternetService
  - OnlineSecurity
  - StreamingTV
  - TechSupport

- Información financiera
  - MonthlyCharges
  - TotalCharges
  - Contract
  - PaymentMethod

- Tiempo de permanencia
  - tenure

### Tipos de datos

El dataset contiene:

- Variables categóricas
- Variables numéricas
- Variables binarias

---

# Problema que Resuelve

Este dataset se utiliza para resolver un problema de:

## Clasificación Binaria

El objetivo es predecir si un cliente abandonará el servicio de telecomunicaciones.

La predicción de churn es importante porque adquirir nuevos clientes suele ser más costoso que conservar los clientes actuales.

---

# Aplicaciones Prácticas

Este tipo de modelo puede utilizarse en escenarios reales como:

## 1. Retención de Clientes

Las empresas pueden identificar clientes con alto riesgo de abandono y ofrecer promociones o descuentos personalizados.

## 2. Marketing Inteligente

Permite crear campañas enfocadas en clientes específicos para reducir pérdidas.

## 3. Optimización de Ingresos

Ayuda a disminuir la pérdida de ingresos causada por cancelaciones de servicios.

## 4. Sistemas de Recomendación

Las compañías pueden recomendar servicios adicionales a clientes con riesgo de churn.

## 5. Análisis Predictivo Empresarial

El modelo puede integrarse en plataformas de Business Intelligence para apoyar la toma de decisiones.

---

# Implicaciones Éticas y Sesgos

## Variables Sensibles

El dataset contiene variables potencialmente sensibles como:

- gender
- SeniorCitizen
- Partner
- Dependents

Estas variables podrían generar sesgos en las predicciones si el modelo aprende patrones discriminatorios.

## Riesgo de Sesgo

Algunos grupos de clientes podrían estar subrepresentados, provocando predicciones menos precisas para ciertos perfiles.

Por ejemplo:

- Adultos mayores
- Personas con contratos específicos
- Usuarios con pocos servicios contratados

## Transparencia del Modelo

Es importante interpretar correctamente las predicciones del modelo y evitar decisiones automáticas injustas.

El modelo debe utilizarse como herramienta de apoyo y no como único criterio para tomar decisiones comerciales.

## Limitaciones

- El dataset representa únicamente clientes de telecomunicaciones.
- No incluye factores externos como satisfacción del cliente o competencia de mercado.
- Los datos podrían no reflejar comportamientos actuales debido al cambio en hábitos digitales.

---

# Conclusión

El dataset Telco Customer Churn es ampliamente utilizado para proyectos de Machine Learning orientados a clasificación y análisis predictivo.

Permite aplicar técnicas de limpieza de datos, entrenamiento de modelos, evaluación de métricas y construcción de pipelines MLOps en un entorno similar al mundo real.