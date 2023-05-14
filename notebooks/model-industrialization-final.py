import pandas as pd
from house_prices.train import build_model
from house_prices.inference import make_predictions

training_data_df = pd.read_csv('/Users/admin-20218/Downloads/house-prices-advanced-regression-techniques/train.csv')
model_performance_dict = build_model(training_data_df)
print(model_performance_dict)

user_data_df = pd.read_csv('/Users/admin-20218/Downloads/house-prices-advanced-regression-techniques/test.csv')
predictions = make_predictions(user_data_df)
print(predictions)

