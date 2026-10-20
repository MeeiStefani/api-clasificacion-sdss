from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el NUEVO modelo supervisado
modelo = joblib.load('modelo_supervisado.pkl')
scaler = joblib.load('scaler.pkl')

class DatosAstro(BaseModel):
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: float

@app.post("/predecir")
def predecir_clase(datos: DatosAstro):
    df_nuevo = pd.DataFrame([datos.model_dump()])
    datos_escalados = scaler.transform(df_nuevo)
    
    # La predicción ahora es un texto exacto ('GALAXY', 'STAR' o 'QSO')
    prediccion_texto = modelo.predict(datos_escalados)[0]
    
    return {
        "mensaje": "Clasificación completada", 
        "clase_predicha": prediccion_texto
    }