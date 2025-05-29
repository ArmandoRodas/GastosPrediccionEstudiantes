import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IA Sanarate", layout="wide")

# — Cabecera —
st.image("Inteligencia-artificial.jpg", use_column_width=True)
st.markdown("<h1 style='text-align:center;'>Inteligencia artificial</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>9no. Ingeniería en Sistemas Sanarate</h3>", unsafe_allow_html=True)
st.write("---")

# — Menú en un expander que puedes colapsar —
with st.sidebar.expander("🔧 Menú", expanded=True):
    seccion = st.radio("Elige la aplicación", ["Predicción de Gastos", "Proyecto Deep Learning"])

@st.cache_resource
def load_expenses_pipeline():
    return joblib.load("models/expenses_model.pkl")

if seccion == "Predicción de Gastos":
    pipeline = load_expenses_pipeline()

    # Obtenemos el ColumnTransformer
    ct = pipeline.named_steps["prep"]
    # Buscamos la tupla cuyo name sea "cat"
    for name, transformer, cols in ct.transformers_:
        if name == "cat":
            ohe = transformer
            cat_cols = cols
            break
    cats = ohe.categories_

    # Layout de dos columnas
    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        st.subheader("🧮 Parámetros de Entrada")
        comidas_fuera = st.number_input("Comidas fuera de casa", min_value=0, step=1)
        snacks_q      = st.number_input("Cantidad de snacks",      min_value=0, step=1)
        edad          = st.number_input("Edad",                    min_value=12, step=1)
        materias_dia  = st.number_input("Cursos en el día",        min_value=0, step=1)
        gasolina_q    = st.number_input("Dinero para gasolina (Q)", min_value=0.0, step=0.1)

        # Generamos los selectboxes dinámicamente
        selections = {}
        for col_name, options in zip(cat_cols, cats):
            label = col_name.replace("_", " ").capitalize()
            selections[col_name] = st.selectbox(label, options)

        if st.button("▶️ Calcular gasto"):
            data = {
                **selections,
                "comidas_fuera": comidas_fuera,
                "snacks_q":      snacks_q,
                "edad":          edad,
                "materias_dia":  materias_dia,
                "gasolina_q":    gasolina_q
            }
            df_in = pd.DataFrame([data])
            gasto = pipeline.predict(df_in)[0]
            st.success(f"💰 Gasto estimado: Q{gasto:.2f}")

    with col2:
        st.subheader("📊 Resultados")
        with st.expander("Ver datos de entrenamiento", expanded=False):
            st.write("Aquí podrías mostrar tu DataFrame de entrenamiento")

else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Próximamente integraré tu modelo de Deep Learning aquí.")
