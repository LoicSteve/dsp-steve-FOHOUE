import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from house_prices.preprocess import preprocess_data


def train_model(data: pd.DataFrame) -> LinearRegression:
    dp = preprocess_data(data)
    y_train = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        dp, y_train, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def ev_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(np.log(y_test), np.log(y_pred)))
    return rmse, rmsle


def build_model(data: pd.DataFrame) -> dict[str, str]:
    model, X_test, y_test = train_model(data)
    rmse, rmsle = ev_model(model, X_test, y_test)
    return {'rmse': str(rmse), 'rmsle': str(rmsle)}
