import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Cargar los datos
df = pd.read_excel("data/datos_gasto_ampliado.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip().str.lower()

# Renombrar columnas
df = df.rename(columns={
    'lugar de origen': 'lugar_de_origen',
    'transporte en el que viaja': 'transporte_en_el_que_viaja',
    'comidas en la uni': 'comidas_en_la_universidad',
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
    'gasto_total_q': 'gasto_total'
})

# Columnas categóricas (❌ sin gasolina)
cat_cols = [
    "lugar_de_origen", "transporte_en_el_que_viaja", "compra_snacks", "actividades_extra",
    "lleva_almuerzo", "compra_almuerzo", "ocupacion", "desayuno_casa",
    "compra_desayuno", "comparte_transporte"
]

# Codificar categóricas
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Columnas numéricas
num_cols = ["comidas_en_la_universidad", "edad", "cursos_dia"]

# Entradas finales
X = df[cat_cols + num_cols]
y = df["gasto_total"]

# Crear pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

# Entrenar y guardar
pipeline.fit(X, y)
joblib.dump(pipeline, "models/expenses_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
print("✔ Modelo actualizado y guardado sin gasolina.")
