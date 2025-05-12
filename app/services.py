import numpy as np
import tensorflow as tf

_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model("app/ml/model.keras") 
    return _model

def make_prediction(input_data):
    try:
        model = get_model()

        input_values = list(input_data.values())
        input_array = np.array(input_values).reshape(1, -1)

        prediction = model.predict(input_array)

        label = int(prediction[0] > 0.8)
        probability = float(prediction[0])

        return {"label": label, "probability": probability}

    except Exception as e:
        return {"error": str(e)}
