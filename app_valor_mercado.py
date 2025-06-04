import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Cargar modelo, scaler y columnas
# -------------------------------
modelo = joblib.load("modelo_rf.pkl")
scaler = joblib.load("scaler.pkl")
columnas = joblib.load("columnas.pkl")

# -------------------------------
# Configuración de página
# -------------------------------
st.set_page_config(page_title="Predicción de Valor de Mercado", layout="centered")

# -------------------------------
# Estilo personalizado
# -------------------------------
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            color: #6c911d;
            font-family: 'Arial', sans-serif;
            font-color:  #6c911d;
        }
        .stButton>button {
            background-color: #6c911d;
            color: white;
        }
        .stDownloadButton>button {
            background-color: #6c911d;
            color: white;
        }

        h1, h2, h3, h4, h5, h6, .stMetric, .stText, .stSubheader, .stHeader {
            color: #6c911d !important;
        }
        .css-1v0mbdj, .css-qri22k, .css-h5rgaw {
            color: ##6c911d !important;
        }
        .main {
            background-color: #FDF6EC;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Encabezado con logo
# -------------------------------
st.image("logo.png", width=120)  # Asegúrate de que "logo.png" está en la misma carpeta
st.title("⚽ Predicción del Valor de Mercado Jugador")

# -------------------------------
# Selección del modo de entrada
# -------------------------------
modo = st.radio("Selecciona modo de entrada:", ["Manual", "Cargar CSV"])

# -------------------------------
# Función de predicción
# -------------------------------
def hacer_prediccion(df_input):
    df_input = df_input[columnas]  # asegurar el orden y que existan las columnas necesarias
    df_scaled = scaler.transform(df_input)
    pred = modelo.predict(df_scaled)
    return np.round(pred, 2)

# -------------------------------
# Entrada Manual
# -------------------------------
if modo == "Manual":
    st.header("📝 Ingreso manual de datos")

    col1, col2 = st.columns(2)

    with col1:
        goles = st.number_input("Goles", min_value=0, value=0)
        asistencias = st.number_input("Asistencias", min_value=0, value=0)
        minutos = st.number_input("Minutos jugados", min_value=0, value=0)
        amarillas = st.number_input("Tarjetas amarillas", min_value=0, value=0)
        rojas = st.number_input("Tarjetas rojas", min_value=0, value=0)
        altura = st.number_input("Altura (en metros)", min_value=1.50, max_value=2.20, value=1.80, step=0.01)
        edad = st.number_input("Edad", min_value=15, max_value=45, value=25)

    with col2:
        part_no_ganados = st.number_input("Partidos no ganados", min_value=0, value=0)
        part_ganados = st.number_input("Partidos ganados", min_value=0, value=0)
        porteria_cero = st.number_input("Porterías a cero", min_value=0, value=0)
        partidos_major = st.number_input("Partidos Major League", min_value=0, value=0)
        partidos_europeos = st.number_input("Partidos Europeos", min_value=0, value=0)
        finales_europeas = st.number_input("Finales Europeas", min_value=0, value=0)
        transferencias = st.number_input("Número de transferencias", min_value=0, value=0)
        lesiones = st.number_input("Número de lesiones", min_value=0, value=0)

    input_data = pd.DataFrame([{
        'goles': goles,
        'asistencias': asistencias,
        'minutos_jugados': minutos,
        'amarillas': amarillas,
        'rojas': rojas,
        'part_no_ganados': part_no_ganados,
        'part_ganados': part_ganados,
        'porteria_cero': porteria_cero,
        'Partidos_majorleague': partidos_major,
        'Partidos_europeos': partidos_europeos,
        'Finales_europeas': finales_europeas,
        'n_transferencias': transferencias,
        'n_lesiones': lesiones,
        'altura': altura,
        'edad': edad
    }])

    if st.button("🔮 Predecir valor de mercado"):
        resultado = hacer_prediccion(input_data)
        st.success(f"💰 Valor estimado del jugador: €{resultado[0]:,.2f}")

# -------------------------------
# Carga CSV
# -------------------------------
else:
    st.header("📂 Cargar CSV")
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        df_csv = pd.read_csv(archivo)

        if not set(columnas).issubset(df_csv.columns):
            st.error("❌ El archivo no contiene todas las columnas necesarias.")
            st.write("Se esperan las siguientes columnas:")
            st.write(columnas)
        else:
            predicciones = hacer_prediccion(df_csv)
            df_csv["valor_mercado_predicho"] = predicciones
            st.success("✅ Predicciones generadas correctamente")
            st.dataframe(df_csv)

            csv = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar resultados", csv, "predicciones.csv", "text/csv")
