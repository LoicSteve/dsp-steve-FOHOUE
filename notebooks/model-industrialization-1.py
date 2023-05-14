import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_squared_log_error

### Model building
## Model training
train_data = pd.read_csv('/Users/admin-20218/Downloads/house-prices-advanced-regression-techniques/train.csv')
X_train = train_data.drop(["Id", "SalePrice"], axis=1)
y_train = train_data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

continuous_features = ["LotArea", "GrLivArea"]
categorical_features = ["MSZoning", "Neighborhood"]

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown="ignore")
scaler.fit(X_train[continuous_features])
X_train[continuous_features] = scaler.transform(X_train[continuous_features])
#X_test[continuous_features] = scaler.transform(X_test[continuous_features])
#X_test[continuous_features] = scaler.transform(X_test[continuous_features])
encoder.fit(X_train[categorical_features])
X_train_processed = encoder.transform(X_train[categorical_features])
#X_test_processed = encoder.transform(X_test[categorical_features])
#X_test_processed = encoder.transform(X_test[categorical_features])

model = LinearRegression()
model.fit(np.hstack((X_train[continuous_features], X_train_processed.toarray())), y_train)

## Model evaluation

# Preprocessing and feature engineering of the test set
X_test[continuous_features] = scaler.transform(X_test[continuous_features])
X_test_processed = encoder.transform(X_test[categorical_features])

# Model predictions on the test set
#y_pred = model.predict(X_test_processed)
y_pred = model.predict(np.hstack((X_test[continuous_features], X_test_processed.toarray())))

def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

#y_pred = model.predict(np.hstack((X_test[continuous_features], X_test_processed.toarray())))
rmse = sqrt(mean_squared_error(y_test, y_pred))
rmsle = compute_rmsle(np.log(y_test), np.log(y_pred))
print("RMSE:", rmse)
print("RMSLE:", rmsle)



###model inference
test_data = pd.read_csv('/Users/admin-20218/Downloads/house-prices-advanced-regression-techniques/test.csv')
X_test = test_data.drop("Id", axis=1)

X_test[continuous_features] = scaler.transform(X_test[continuous_features])
X_test_processed = encoder.transform(X_test[categorical_features])

# Model predictions on the test set
predictions = model.predict(np.hstack((X_test[continuous_features], X_test_processed.toarray())))
#y_pred = model.predict(np.hstack((X_test[continuous_features], X_test_processed.toarray())))

# Model evaluation
output = pd.DataFrame({"Id": test_data["Id"], "SalePrice": predictions})
print(output)