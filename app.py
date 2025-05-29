import streamlit as st
import pandas as pd
import joblib

# ——————————————————————————————————————————————
# CABECERA GLOBAL
# ——————————————————————————————————————————————

# Muestra la imagen del proyecto
st.image("Inteligencia-artificial.jpg", use_column_width=True)

# Títulos arriba de todo
st.markdown("<h1 style='text-align:center;'>Inteligencia artificial</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>9no. Ingeniería en Sistemas Sanarate</h3>", unsafe_allow_html=True)
st.write("---")  # línea dividora

# ——————————————————————————————————————————————
# MENÚ LATERAL
# ——————————————————————————————————————————————
st.sidebar.title("🔧 Menú")
seccion = st.sidebar.radio(
    "Elige la aplicación",
    ["Predicción de Gastos", "Proyecto Deep Learning"]
)

# ——————————————————————————————————————————————
# FUNCIONES AUXILIARES
# ——————————————————————————————————————————————
@st.cache_resource
def load_expenses_pipeline():
    return joblib.load("models/expenses_pipeline.pkl")

# (Más adelante, podrías cargar tu modelo de DL en otra función)


# ——————————————————————————————————————————————
# SECCIÓN 1: Predicción de Gastos
# ——————————————————————————————————————————————
if seccion == "Predicción de Gastos":
    st.header("📊 Predicción de gastos estudiantiles")
    # Carga el pipeline entrenado
    pipeline = load_expenses_pipeline()

    # Entradas numéricas
    comidas_fuera = st.number_input("Comidas fuera de casa", min_value=0, step=1)
    snacks_q      = st.number_input("Cantidad de snacks",      min_value=0, step=1)
    edad          = st.number_input("Edad",                    min_value=12, step=1)
    materias_dia  = st.number_input("Cursos en el día",        min_value=0, step=1)
    gasolina_q    = st.number_input("Dinero para gasolina (Q)", min_value=0.0, step=0.1)

    # Extrae dinámicamente las categorías de tu OHE
    ohe = pipeline.named_steps["prep"].transformers_[1][1]
    cats = ohe.categories_
    lugar_opts       = list(cats[0])
    transporte_opts  = list(cats[1])
    act_ext_opts     = list(cats[2])
    almuerzo_opts    = list(cats[3])
    compra_alm_opts  = list(cats[4])
    ocup_opts        = list(cats[5])
    desayuno_opts    = list(cats[6])
    compra_des_opts  = list(cats[7])
    comparte_opts    = list(cats[8])

    # Entradas categóricas
    lugar            = st.selectbox("Lugar de origen",       lugar_opts)
    transporte       = st.selectbox("Transporte",            transporte_opts)
    actividades_extra= st.selectbox("Actividades extra",     act_ext_opts)
    lleva_almuerzo   = st.selectbox("Lleva almuerzo",        almuerzo_opts)
    compra_almuerzo  = st.selectbox("Compra almuerzo",       compra_alm_opts)
    ocupacion        = st.selectbox("Ocupación",             ocup_opts)
    desayuno_casa    = st.selectbox("Desayuno en casa",      desayuno_opts)
    compra_desayuno  = st.selectbox("Compra desayuno",       compra_des_opts)
    comparte_transp  = st.selectbox("Comparte transporte",   comparte_opts)

    if st.button("▶️ Calcular gasto"):
        df_in = pd.DataFrame([{
            "lugar":               lugar,
            "transporte":          transporte,
            "actividades_extra":   actividades_extra,
            "lleva_almuerzo":      lleva_almuerzo,
            "compra_almuerzo":     compra_almuerzo,
            "ocupacion":           ocupacion,
            "desayuno_casa":       desayuno_casa,
            "compra_desayuno":     compra_desayuno,
            "comparte_transporte": comparte_transp,
            "comidas_fuera":       comidas_fuera,
            "snacks_q":            snacks_q,
            "edad":                edad,
            "materias_dia":        materias_dia,
            "gasolina_q":          gasolina_q
        }])
        gasto = pipeline.predict(df_in)[0]
        st.success(f"💰 Gasto estimado: Q{gasto:.2f}")

# ——————————————————————————————————————————————
# SECCIÓN 2: Proyecto Deep Learning (placeholder)
# ——————————————————————————————————————————————
else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Esta sección estará disponible cuando integres tu segundo modelo de DL.")
