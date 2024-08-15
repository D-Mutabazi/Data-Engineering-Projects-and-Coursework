import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Load and preprocess data
df = pd.read_csv('./Data/EURUSD_D1.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

X = df[['Open', 'High', 'Low', 'Volume']]  # Features
Y = df['Close']  # Target

# Split into training and test (80/20 split) using train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Train linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Predict on test data
pred = regr.predict(X_test)

# Evaluate Model Performance
mse = mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
print("Model Performance Metrics: \n")
print("RMSE: ", str(rmse))
print("MSE: ", str(mse))
print("MAE: ", str(mae))

# Plotting
test_dates = df['Time'].iloc[-len(y_test):]  # Use the last part of dates for the test set

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, color='b', label='True Price')
plt.plot(test_dates, pred, color='r', label='Predicted')
plt.xlabel('Dates')
plt.ylabel('Price')
plt.title("True and Predicted Test Set Results")
plt.legend()
plt.show()
