# ⚽ Cazatalentos – Predicción del Valor de Mercado de Futbolistas

**Cazatalentos** es una aplicación desarrollada como proyecto final de un bootcamp de analista de datos. Su objetivo es predecir el valor de mercado de jugadores de fútbol a partir de características específicas del jugador, utilizando técnicas de machine learning.

---

## 📌 Descripción del Proyecto

Durante el desarrollo de este proyecto, se puso un gran énfasis en todo el proceso de ciencia de datos, desde la obtención y limpieza de los datos hasta la creación de un modelo predictivo funcional.

### 🔍 Búsqueda y Selección de Datos

Se exploraron múltiples fuentes de datos públicas relacionadas con el rendimiento y características de jugadores de fútbol profesionales. La selección de los datasets adecuados fue una etapa crítica para garantizar la calidad del modelo.

### 🧼 Limpieza y Preparación de los Datos

La parte más extensa del trabajo se enfocó en la preparación del dataset:

- Tratamiento de valores nulos
- Estandarización y normalización de variables
- Ingeniería de características
- Eliminación de outliers y duplicados

Este proceso permitió construir un conjunto de datos robusto y consistente para el modelado.

### 📊 Visualización y Toma de Decisiones

Se realizaron múltiples visualizaciones con `matplotlib`, `seaborn` y `plotly` para entender mejor las relaciones entre variables, identificar patrones y tomar decisiones clave en la selección de variables para el modelo.

### 🤖 Modelado con Machine Learning

Finalmente, se entrenó un modelo de **Random Forest Regressor** utilizando `scikit-learn`, logrando buenos resultados en la predicción del valor de mercado.

---

## 🎯 Funcionalidad de la App

La app, desarrollada con `Streamlit`, permite al usuario ingresar los atributos de un jugador (edad, posición, estadísticas, etc.) y devuelve una predicción de su valor estimado en el mercado actual.

---

## 🧪 Tecnologías Usadas

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- Joblib
- Streamlit

---

## 🚀 Cómo usar

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/cazatalentos.git

## 👨‍💻 Autor
Este proyecto fue realizado como trabajo final de un bootcamp de Analisis de datos por:
- Giovanny Rodriguez
- Adriá Gras

##  📎 Demo en línea
Puedes probar la app aquí:
👉 https://cazatalentos.streamlit.app/
