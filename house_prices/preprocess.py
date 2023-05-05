import numpy as np
import pandas as pd
import joblib


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    encoder = joblib.load('../models/encoder.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    continuous_features = ["LotArea", "GrLivArea"]
    categorical_features = ["MSZoning", "Neighborhood"]
    data[continuous_features] = scaler.transform(data[continuous_features])
    data_processed = encoder.transform(data[categorical_features])

    return np.hstack((data[continuous_features], data_processed.toarray()))
