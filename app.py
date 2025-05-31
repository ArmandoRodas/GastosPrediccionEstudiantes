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

# Cargar modelo
@st.cache_resource
def load_pipeline():
    return joblib.load("models/expenses_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load("models/label_encoders.pkl")

# SECCIÓN 1: Predicción de Gastos
if seccion == "Predicción de Gastos":
    try:
        pipeline = load_pipeline()
        label_encoders = load_encoders()

        ct = pipeline.named_steps["prep"]
        ohe = None
        cat_cols = []

        for name, transformer, cols in ct.transformers_:
            if name == "cat":
                ohe = transformer
                cat_cols = cols
                break

        if ohe is None:
            st.error("Error: No se encontró codificador categórico en el pipeline.")
            st.stop()

        cats = ohe.categories_

        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.subheader("🧮 Parámetros de Entrada")

            # Entradas numéricas
            comidas_universidad = st.number_input("Comidas en la Universidad", min_value=0, step=1)
            edad = st.number_input("¿Tu Edad?", min_value=18, step=1)
            cursos_dia = st.number_input("¿Por cuántos Cursos vas en el día de Universidad?", min_value=0, step=1)

            # Entradas categóricas
            selections = {}
            for col_name, options in zip(cat_cols, cats):
                label = col_name.replace("_", " ").capitalize()
                selections[col_name] = st.selectbox(label, options)

            if st.button("▶️ Calcular gasto"):
                data = {
                    **selections,
                    "comidas_universidad": comidas_universidad,
                    "edad": edad,
                    "cursos_dia": cursos_dia
                }

                df_input = pd.DataFrame([data])

                # Aplicar los LabelEncoders para convertir a numérico
                for col, le in label_encoders.items():
                    if col in df_input:
                        df_input[col] = le.transform(df_input[col])

                pred = pipeline.predict(df_input)[0]
                st.success(f"💰 Gasto estimado: Q{pred:.2f}")

        with col2:
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

                **Modelo utilizado:** Regresión RidgeCV con codificación categórica y escalado numérico.

                Esta predicción ayuda a anticipar gastos semanales y analizar patrones de consumo estudiantil.

                **Streamlit** es la plataforma utilizada para subir a la nube la aplicación.
                Con solo registrar su correo enlazado con **GitHub** (correo de la universidad)
                le permite subir sus repositorios con sus proyectos a la nube y generar su propio enlace.
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error al cargar el pipeline: {e}")

# SECCIÓN 2: Proyecto Deep Learning (Placeholder)
else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Próximamente se integrará aquí el modelo basado en redes neuronales.")
