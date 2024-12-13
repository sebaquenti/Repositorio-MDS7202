from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin



class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.columns_to_drop, errors='ignore')


with open("model/xgboost_etapa3.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = None, input_data: str = Form(None)):
    if file:
        df = pd.read_csv(file.file)
        predictions = model.predict(df)
        return JSONResponse(content={"predictions": predictions.tolist()})
    elif input_data:
        input_dict = eval(input_data)  
        df = pd.DataFrame([input_dict])
        prediction = model.predict(df)
        return JSONResponse(content={"prediction": prediction[0]})
    else:
        return JSONResponse(content={"error": "No se recibió entrada válida"}, status_code=400)
