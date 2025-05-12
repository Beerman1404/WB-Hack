from fastapi import FastAPI, HTTPException
from app.services import get_model, make_prediction
from app.schemas import InputData, PredictionResult

app = FastAPI()

# Загрузка модели
model = get_model()

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData):
    try:
        # Подготовка данных для модели (конвертирование в список или массив)
        input_data = [data.dict().values()]
        
        # Получение предсказания
        prediction = make_prediction(model, input_data)
        
        if prediction is None:
            raise HTTPException(status_code=500, detail="Ошибка предсказания")
        
        label = prediction[0]  # Предсказанная метка
        probability = prediction[1]  # Вероятность (если есть)

        return PredictionResult(label=label, probability=probability)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
