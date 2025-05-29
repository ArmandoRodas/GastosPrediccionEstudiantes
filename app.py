import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
#  Configuración de página
# ---------------------------------------------------
st.set_page_config(
    page_title="IA Sanarate", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
#  Cabecera con imagen y títulos
# ---------------------------------------------------
st.image("Inteligencia-artificial.jpg", use_column_width=True)
st.markdown("<h1 style='text-align:center;'>Inteligencia artificial</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>9no. Ingeniería en Sistemas Sanarate</h3>", unsafe_allow_html=True)
st.write("---")

# ---------------------------------------------------
#  Sidebar: menú de selección y luego desaparecer
# ---------------------------------------------------
if 'app_selected' not in st.session_state:
    st.session_state.app_selected = st.sidebar.radio(
        "🔧 Menú", 
        ["Predicción de Gastos", "Proyecto Deep Learning"]
    )
    # Una vez elegido, vaciamos la sidebar
    st.sidebar.empty()

# Guarda qué sección debemos mostrar
seccion = st.session_state.app_selected

# ---------------------------------------------------
#  Función para cargar tu pipeline de gastos
# ---------------------------------------------------
@st.cache_resource
def load_expenses_pipeline():
    return joblib.load("models/expenses_model.pkl")

# ---------------------------------------------------
#  Sección 1: Predicción de Gastos
# ---------------------------------------------------
if seccion == "Predicción de Gastos":
    pipeline = load_expenses_pipeline()
    
    # Contenedor de dos columnas: izquierda inputs, derecha resultados
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.subheader("🧮 Parámetros de Entrada")
        comidas_fuera = st.number_input("Comidas fuera de casa", min_value=0, step=1)
        snacks_q      = st.number_input("Cantidad de snacks",      min_value=0, step=1)
        edad          = st.number_input("Edad",                    min_value=12, step=1)
        materias_dia  = st.number_input("Cursos en el día",        min_value=0, step=1)
        gasolina_q    = st.number_input("Dinero para gasolina (Q)", min_value=0.0, step=0.1)
        
        # Extraemos las categorías desde el propio pipeline
        ohe = pipeline.named_steps["prep"].transformers_[1][1]
        cat_cols = pipeline.named_steps["prep"].transformers_[1][2]
        cats     = ohe.categories_
        
        selections = {}
        for col_name, options in zip(cat_cols, cats):
            label = col_name.replace("_", " ").capitalize()
            selections[col_name] = st.selectbox(label, options)
        
        # Botón de predicción
        if st.button("▶️ Calcular gasto"):
            data = {**selections,
                    "comidas_fuera": comidas_fuera,
                    "snacks_q":      snacks_q,
                    "edad":          edad,
                    "materias_dia":  materias_dia,
                    "gasolina_q":    gasolina_q}
            df_in = pd.DataFrame([data])
            gasto = pipeline.predict(df_in)[0]
            st.success(f"💰 Gasto estimado: Q{gasto:.2f}")
    
    with col2:
        st.subheader("📊 Resultados")
        # Aquí podrías mostrar gráficas o métricas adicionales
        # Por ejemplo:
        # st.metric("MAE entreno", "Q2.43")
        # Expander con datos de entrenamiento
        with st.expander("Ver datos de entrenamiento", expanded=False):
            # Asumiendo que cargaste df en tu pipeline o lo tienes disponible:
            # df = pd.read_csv("data/tu_datos.csv")
            # st.dataframe(df)
            st.write("Aquí van los datos de tu set de entrenamiento")

# ---------------------------------------------------
#  Sección 2: Placeholder Deep Learning
# ---------------------------------------------------
else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Próximamente integraré tu modelo de Deep Learning aquí.")
