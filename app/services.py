from app.ml.model import load_model, predict

# Загрузка модели
def get_model():
    model = load_model()
    return model

# Получение предсказания
def make_prediction(model, input_data):
    prediction = predict(model, input_data)
    return prediction
