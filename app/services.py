import pandas as pd
from app.ml import model

def predict(input_dict: dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)

