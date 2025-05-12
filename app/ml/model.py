import tensorflow as tf

# Загружаем модель
def load_model():
    try:
        # Загрузка модели TensorFlow (Keras)
        model = tf.keras.models.load_model("app/ml/model.keras")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

# Функция для предсказания
def predict(model, input_data):
    try:
        # Убедимся, что модель загружена
        if model is None:
            raise ValueError("Модель не загружена!")

        # Подготовка входных данных
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Предсказание
        prediction = model.predict(input_data)
        
        # Возвращаем результат
        return prediction
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None

