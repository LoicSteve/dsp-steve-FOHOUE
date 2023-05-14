import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from house_prices.preprocess import preprocess_data_train


def train_model(data: pd.DataFrame) -> LinearRegression:
    X_tr, y_train, X_t, y_test = preprocess_data_train(data)
    y_train = data["SalePrice"]
    model = LinearRegression()
    model.fit(X_tr, y_train)
    joblib.dump(model, '../models/model.joblib')
    return model, X_t, y_test


def ev_model(model: LinearRegression, X_t: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_t)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(np.log(y_test), np.log(y_pred)))
    return rmse, rmsle


def build_model(data: pd.DataFrame) -> dict[str, str]:
    model, X_t, y_test = train_model(data)
    rmse, rmsle = ev_model(model, X_t, y_test)
    return {'rmse': str(rmse), 'rmsle': str(rmsle)}
