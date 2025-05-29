import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
import joblib

# 1) Carga datos
df = pd.read_excel("data/datos_gasto_ampliado.xlsx", engine="openpyxl")

# 2) Normaliza nombres de columnas y renombra
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    'lugar de origen':                  'lugar',
    'transporte en el que viaja':       'transporte',
    'comidas en la uni':                'comidas_uni',         # numérico
    'compra snacks':                    'compra_snacks',      # sí/no
    'actividades extra en la uni':      'actividades_extra',
    'lleva almuerzo':                   'lleva_almuerzo',
    'compra almuerzo':                  'compra_almuerzo',
    'ocupacion':                        'ocupacion',
    'edad':                             'edad',
    'cursos en el dia':                 'cursos_dia',          # numérico
    'desayuno en casa':                 'desayuno_casa',
    'compra desayuno':                  'compra_desayuno',
    'comparte transporte':              'comparte_transporte',
    'hecha o da dinero para gasolina':  'gasolina_q',          # numérico
    'gasto_total_q':                    'gasto_total'
})

# Asegúrate de que la salida sea int
df['gasto_total'] = df['gasto_total'].astype(int)

# 3) Define variables categóricas y numéricas
cat_cols = [
    "lugar",
    "transporte",
    "compra_snacks",       # ahora es categórica
    "actividades_extra",
    "lleva_almuerzo",
    "compra_almuerzo",
    "ocupacion",
    "desayuno_casa",
    "compra_desayuno",
    "comparte_transporte"
]
num_cols = [
    "comidas_uni",         # numérico
    "edad",
    "cursos_dia",
    "gasolina_q"           # numérico
]

X = df[cat_cols + num_cols]
y = df["gasto_total"]

# 4) Monta el pipeline con paso "prep" y "reg"
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),    num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
])
pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg",  RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

# 5) Entrena y serializa
pipeline.fit(X, y)
joblib.dump(pipeline, "models/expenses_pipeline.pkl")
print("✔ Pipeline entrenado y guardado en models/expenses_pipeline.pkl")
