import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Import dataset
df = pd.read_csv('./Data/EURUSD_D1.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

print(df.info())

# Feature and Target Extraction
X = df[['Open', 'High', 'Low']]
Y = df['Close']

print(X.head(), Y.head())

# Test and training split (80/20)
shape_80 = int(X.shape[0] * 0.8)

x_train = X[:shape_80]
y_train = Y[:shape_80]

x_test = X[shape_80:]
y_test = Y[shape_80:]
test_dates = df['Time'][shape_80:]

# Model training and fit
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
regressor.fit(x_train, y_train)

# Model Predictions and Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# OOB score
oob_score = regressor.oob_score_
print(f'OOB Score: {oob_score}')

# Prediction on test
predictions = regressor.predict(x_test)
predictions = pd.DataFrame(predictions, columns=['Predicted Price'])

mse = mean_squared_error(y_test, predictions)
print(f'Mean squared error: {mse}')

# Plotting
predictions.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

combined = pd.concat([test_dates, predictions], axis=1)

y_test = y_test.reset_index(drop=True)
y_test.name = 'Close'
testData_combined = pd.concat([test_dates, y_test], axis=1)

plt.plot(testData_combined['Time'], testData_combined['Close'], color='b', label='True Price')
plt.plot(combined['Time'], combined['Predicted Price'], color='r', label='Predicted')

plt.xlabel('Dates')
plt.ylabel('Price')
plt.title("True and Predicted Test Set Results")
plt.legend()

plt.show()
