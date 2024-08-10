import matplotlib.pyplot as plt
import pandas as pd                 #pandas are used to analyze data
import plotly.graph_objects as go   #plot candle stick data - plotly library
from sklearn import linear_model    # multiple linear regression model
import math                         # evaluate model performances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('./Data/EURUSD_H1.csv')
df['Time'] = pd.to_datetime(df['Time'],format='%Y-%m-%d %H:%M:%S')

X = df[['Open','High','Low']]  #features
Y = df['Close']                #target

#split into training and test (80/20 split)
shape_80 = int(X.shape[0]*0.8)-1

x_train = X[:shape_80]
y_train = Y[:shape_80]
train_date = df['Time'][:shape_80]
print(train_date.head())

x_test = X[shape_80:]
y_test = Y[shape_80:]
test_dates = df['Time'][shape_80:]
test_dates = pd.to_datetime(test_dates)
test_dates = pd.DataFrame(test_dates)

#train linear regression
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train) 

#predict on test data
pred = regr.predict(x_test)
pred = pd.DataFrame(pred, columns=['Predicted Price']) # convert numpy array to dataframe

#Reset the index of the predictions to align with the test dates
pred.reset_index(drop=True, inplace=True)
test_dates.reset_index(drop=True, inplace=True)

combined = pd.concat([test_dates, pred], axis =1 )

#plot test data
y_test.reset_index(drop=True, inplace=True)
testData_combined = pd.concat([test_dates, y_test], axis=1)

plt.plot(testData_combined['Time'],testData_combined['Close'], color='b', label='True Price')
plt.plot(combined['Time'],combined['Predicted Price'], color='r', label='Predicted')

plt.xlabel('Dates')
plt.ylabel('Price')
plt.title("True and Predicted test set results")
plt.legend()

plt.show()

#Evaluate Model Performance
mse = mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
print("Model Performance Metrics: \n")
print("RMSE: ",str(rmse))
print("MSE: ",str(mse))
print("MAE: ",str(mae))




