import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IA Sanarate", layout="wide")

# — Cabecera —
st.image("Inteligencia-artificial.jpg", use_column_width=True)
st.markdown("<h1 style='text-align:center;'>Inteligencia artificial</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>9no. Ingeniería en Sistemas Sanarate</h3>", unsafe_allow_html=True)
st.write("---")

# — Menú lateral —
with st.sidebar.expander("🔧 Menú", expanded=True):
    seccion = st.radio("Elige la aplicación", ["Predicción de Gastos", "Proyecto Deep Learning"])

@st.cache_resource
def load_pipeline():
    return joblib.load("models/expenses_model.pkl")  # ⚠️ Asegúrate de que sea este archivo

if seccion == "Predicción de Gastos":
    pipeline = load_pipeline()

    # ✅ CORREGIDO: acceso directo
    ct = pipeline.named_steps["prep"]
    for name, transformer, cols in ct.transformers_:
        if name == "cat":
            ohe = transformer
            cat_cols = cols
            break
    cats = ohe.categories_

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("🧮 Parámetros de Entrada")
        comidas_uni = st.number_input("Comidas en la Uni", min_value=0, step=1)
        edad = st.number_input("Edad", min_value=12, step=1)
        cursos_dia = st.number_input("Cursos en el día", min_value=0, step=1)
        gasolina_q = st.number_input("Dinero para gasolina (Q)", min_value=0.0, step=0.1)

        # Selectboxes dinámicos para variables categóricas
        selections = {}
        for col_name, options in zip(cat_cols, cats):
            label = col_name.replace("_", " ").capitalize()
            selections[col_name] = st.selectbox(label, options)

        if st.button("▶️ Calcular gasto"):
            data = {
                **selections,
                "comidas_uni": comidas_uni,
                "edad": edad,
                "cursos_dia": cursos_dia,
                "gasolina_q": gasolina_q
            }
            df_input = pd.DataFrame([data])
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

                **Modelo utilizado:** Regresión RidgeCV con codificación categórica y escalado numérico.

                La predicción ayuda a anticipar gastos semanales y analizar patrones de consumo estudiantil.
            """, unsafe_allow_html=True)

        with st.expander("📂 Ver datos de entrenamiento"):
            st.info("Puedes cargar y mostrar tu DataFrame aquí si lo deseas.")

else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Próximamente se integrará aquí el modelo basado en redes neuronales.")
