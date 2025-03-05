here is the code you can use your own file path



import pandas as pd
import numpy as np

file_path = r"C:\Users\Desktop\sales_data_sample.csv"

x = pd.read_csv(file_path, encoding="ISO-8859-1")

features = ['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'YEAR_ID']
target = 'SALES'

x[features] = x[features].fillna(x[features].mean())

X = x[features]
y = x[target]

X = X.to_numpy()
y = y.to_numpy()

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Add bias term (column of ones)
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Compute the parameters (theta) using Normal Equation
theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

# Make predictions
y_pred = X_test_bias @ theta

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_pred) ** 2)

# Calculate R² Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
