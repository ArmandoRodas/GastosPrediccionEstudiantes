import streamlit as st
import pandas as pd
import joblib

# Configuración de página
st.set_page_config(page_title="IA Sanarate", layout="wide")

# Cabecera
st.image("assets/Inteligencia-artificial.jpg", use_column_width=True)
st.markdown("<h1 style='text-align:center;'>Inteligencia artificial</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>9no. Ingeniería en Sistemas Sanarate</h3>", unsafe_allow_html=True)
st.write("---")

# Menú lateral
with st.sidebar.expander("🔧 Menú", expanded=True):
    seccion = st.radio("Elige la aplicación", ["Predicción de Gastos", "Proyecto Deep Learning"])

# Función para cargar modelo y codificadores
@st.cache_resource
def load_assets():
    model = joblib.load("models/expenses_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    return model, encoders

# Sección de predicción
if seccion == "Predicción de Gastos":
    try:
        pipeline, label_encoders = load_assets()

        st.subheader("🧮 Parámetros de Entrada")

        # Entradas numéricas
        comidas = st.number_input("Comidas en la Universidad", min_value=0, step=1)
        edad = st.number_input("Edad", min_value=18, step=1)
        cursos = st.number_input("Cursos en el Día", min_value=0, step=1)

        # Entradas categóricas
        data_input = {
            "lugar_de_origen": st.selectbox("Lugar de Origen", ["Sansare", "Jalapa", "Guatemala", "Guastatoya", "Sanarate", "Agua Caliente", "San Antonio La Paz"]),
            "transporte_en_el_que_viaja": st.selectbox("Transporte en el que Viaja", ["Bus", "Moto", "Carro", "A pie"]),
            "compra_snacks": st.selectbox("Compra Snacks", ["Sí", "No"]),
            "actividades_extra": st.selectbox("Actividades Extra en la Uni", ["Sí", "No"]),
            "lleva_almuerzo": st.selectbox("Lleva Almuerzo", ["Sí", "No"]),
            "compra_almuerzo": st.selectbox("Compra Almuerzo", ["Sí", "No"]),
            "ocupacion": st.selectbox("Ocupación", ["Estudia", "Trabaja", "Ambas"]),
            "desayuno_casa": st.selectbox("Desayuno en Casa", ["Sí", "No"]),
            "compra_desayuno": st.selectbox("Compra Desayuno", ["Sí", "No"]),
            "comparte_transporte": st.selectbox("Comparte Transporte", ["Sí", "No"]),
            "hecha_o_da_dinero_para_gasolina": st.selectbox("¿Hecha o da dinero para gasolina?", ["Sí", "No"]),
            "comidas_en_la_universidad": comidas,
            "edad": edad,
            "cursos_dia": cursos
        }

        if st.button("▶️ Calcular gasto"):
            df_input = pd.DataFrame([data_input])

            # Aplicar codificadores
            for col, encoder in label_encoders.items():
                if col in df_input.columns:
                    df_input[col] = encoder.transform(df_input[col])

            # Predicción
            pred = pipeline.predict(df_input)[0]
            st.success(f"💰 Gasto estimado: Q{pred:.2f}")

        # Panel lateral
        st.subheader("📊 Resultados")
        with st.expander("📄 Información del Proyecto"):
            st.markdown("""
                Este proyecto predice cuánto gasta un estudiante universitario el día **domingo** cuando asiste a clases.

                **Datos considerados:**
                - Lugar de origen
                - Medio de transporte
                - Snacks, comidas, desayuno
                - Edad, cursos del día, gasolina
                - Ocupación (trabaja, estudia, ambas)
                - Comparte transporte 
                - Hecha o da dinero para gasolina

                **Modelo utilizado:** Regresión RidgeCV con codificación categórica (`LabelEncoder`) y escalado numérico.

                **Plataforma:** Streamlit Cloud + GitHub (correo universitario).
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error al cargar el pipeline: {e}")

# Sección placeholder
else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Próximamente se integrará aquí el modelo basado en redes neuronales.")
