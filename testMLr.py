#  - - - - - CODE PRODUCED BY CHATGPT - - - - - - #

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and prepare the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Time'] = pd.to_datetime(data['Time'],format='%Y-%m-%d %H:%M:%S')

    # Assuming columns 'feature1', 'feature2', ..., 'featureN', and 'target'
    time = data['Time']
    X = data[['Open','High', 'Low']]  #features
    Y = data['Close']                #target

    return X, Y, time

# Define the Linear Regression model
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Calculate predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Main function to run the model
def main():
    # Load data
    file_path = './Data/EURUSD_H1.csv'  # Replace with your CSV file path
    X, y, time = load_data(file_path)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(X, y, time,  test_size=0.2, random_state=42, shuffle=False)

    print(X_train.head())


   # Standardize features - unless volume added, not needed
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Initialize and train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    print(f'Training MSE: {train_mse}')
    print(f'Test MSE: {test_mse}')

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(time_test.values, y_test, color='blue', label='True Values')
    plt.plot(time_test.values, y_pred_test, color='red', linestyle='--', label='Predictions')
    plt.xlabel('Time')
    plt.ylabel('Target')
    plt.title('True Values vs Predictions')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
