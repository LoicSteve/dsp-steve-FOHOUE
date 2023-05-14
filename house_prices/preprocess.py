import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess_data_train(data: pd.DataFrame) -> pd.DataFrame:
    # encoder = joblib.load('../models/encoder.joblib')
    # scaler = joblib.load('../models/scaler.joblib')
    X_train = data.drop(["Id", "SalePrice"], axis=1)
    y_train = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    continuous_features = ["LotArea", "GrLivArea"]
    categorical_features = ["MSZoning", "Neighborhood"]
    encoder = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()
    scaler.fit(X_train[continuous_features])
    encoder.fit(X_train[categorical_features])
    data[continuous_features] = scaler.transform(data[continuous_features])
    data_processed = encoder.transform(data[categorical_features])
    joblib.dump(scaler, "../models/scaler.joblib")
    joblib.dump(encoder, '../models/encoder.joblib')
    X_tr = np.hstack((data[continuous_features], data_processed.toarray()))
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])
    X_test_processed = encoder.transform(X_test[categorical_features])
    X_t = np.hstack((X_test[continuous_features], X_test_processed.toarray()))

    return X_tr, y_train, X_t, y_test


def preprocess_data_test(data: pd.DataFrame) -> pd.DataFrame:
    encoder = joblib.load('../models/encoder.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    continuous_features = ["LotArea", "GrLivArea"]
    categorical_features = ["MSZoning", "Neighborhood"]
    data[continuous_features] = scaler.transform(data[continuous_features])
    data_processed = encoder.transform(data[categorical_features])

    return np.hstack((data[continuous_features], data_processed.toarray()))
