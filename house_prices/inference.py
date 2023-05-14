import pandas as pd
import numpy as np
import joblib
from house_prices.preprocess import preprocess_data_test


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    model = joblib.load('../models/model.joblib')
    input_data_processed = preprocess_data_test(input_data)
    predictions = model.predict(input_data_processed)

    return predictions
