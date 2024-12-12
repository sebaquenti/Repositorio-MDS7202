from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List

app = FastAPI()

# Cargar el modelo
model = joblib.load("xgboost_etapa3.pkl")

class PredictionRequest(BaseModel):
    data: List[dict]

@app.post("/predict/")
async def predict(request: PredictionRequest):
    df = pd.DataFrame(request.data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

