import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
import joblib

# Cargar y preparar los datos
df = pd.read_excel("data/datos_gasto_ampliado.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip().str.lower()

df = df.rename(columns={
    'lugar de origen': 'lugar',
    'transporte en el que viaja': 'transporte',
    'comidas en la uni': 'comidas_uni',
    'compra snacks': 'compra_snacks',
    'actividades extra en la uni': 'actividades_extra',
    'lleva almuerzo': 'lleva_almuerzo',
    'compra almuerzo': 'compra_almuerzo',
    'ocupacion': 'ocupacion',
    'edad': 'edad',
    'cursos en el dia': 'cursos_dia',
    'desayuno en casa': 'desayuno_casa',
    'compra desayuno': 'compra_desayuno',
    'comparte transporte': 'comparte_transporte',
    'hecha o da dinero para gasolina': 'gasolina_q',
    'gasto_total_q': 'gasto_total'
})

# Forzar tipo string en las categóricas
cat_cols = [
    "lugar", "transporte", "compra_snacks", "actividades_extra",
    "lleva_almuerzo", "compra_almuerzo", "ocupacion",
    "desayuno_casa", "compra_desayuno", "comparte_transporte"
]
df[cat_cols] = df[cat_cols].astype(str)

num_cols = ["comidas_uni", "edad", "cursos_dia", "gasolina_q"]
X = df[cat_cols + num_cols]
y = df["gasto_total"]

# Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
])
pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "models/expenses_model.pkl")
print("✔ Modelo entrenado y guardado correctamente")
