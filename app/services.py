import numpy as np

def make_prediction(input_data):
    try:
        # Извлекаем все значения в список (или numpy массив)
        input_values = list(input_data.values())

        # Преобразуем в numpy массив (или другой подходящий формат для модели)
        input_array = np.array(input_values).reshape(1, -1)  # reshape для 2D массива, если требуется

        # Предсказание
        prediction = _model.predict(input_array)

        # Предположим, что модель возвращает метку и вероятность
        label = int(prediction[0] > 0.5)  # Пример, если предсказание вероятностное
        probability = float(prediction[0])

        return {"label": label, "probability": probability}

    except Exception as e:
        return {"error": str(e)}
