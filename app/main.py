from fastapi import FastAPI
from app.schemas import InputData, PredictionResult
from app import services

app = FastAPI()

@app.post("/predict", response_model=PredictionResult)
def predict(input_data: InputData):
    label, prob = services.predict(input_data.dict())
    return PredictionResult(label=label, probability=prob)
