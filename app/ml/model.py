# app/ml/model.py

import numpy as np
from keras.models import load_model

# Загрузка модели .keras (формат Keras v3)
_model = load_model("app/ml/model.keras", safe_mode=False)  # safe_mode=False отключает строгую проверку окружения

def predict(input_data: list[float]) -> tuple[int, float]:
    """
    input_data — список числовых признаков в том же порядке, как во время обучения
    Возвращает кортеж (метка, вероятность)
    """
    input_array = np.array([input_data], dtype=np.float32)  # приведение к нужной форме (1, N)
    probabilities = _model.predict(input_array)[0]
    
    predicted_label = int(np.argmax(probabilities))  # предполагается многоклассовая модель
    confidence = float(np.max(probabilities))        # вероятность предсказания

    return predicted_label, confidence

