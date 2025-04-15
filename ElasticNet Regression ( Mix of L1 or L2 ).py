

import numpy as np
import pandas as pd

file_path = r"----------------------------------"
data = pd.read_csv(file_path, encoding="ISO-8859-1")

features = ['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'YEAR_ID']
target = 'SALES'

data[features] = data[features].fillna(data[features].mean())

X = data[features].to_numpy()
y = data[target].to_numpy()

# Normalize features for stable training
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train-test split (80-20)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Add bias term (column of ones)
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Initialize parameters
lambda_total = 0.1  # Overall regularization strength
l1_ratio = 0.5  # Elastic Net mix (0 = Ridge, 1 = Lasso)
alpha = 0.01  # Learning rate
n_iterations = 1000  # Number of iterations
m, n_features = X_train_bias.shape

theta = np.zeros(n_features)  # Initialize weights

# Elastic Net Gradient Descent
for i in range(n_iterations):
    y_pred = X_train_bias @ theta  # Compute predictions
    error = y_pred - y_train  # Compute error
    
    # Compute gradient of MSE loss
    gradient = (2/m) * (X_train_bias.T @ error)

    # Elastic Net Regularization
    l1_term = lambda_total * l1_ratio * np.sign(theta[1:])  # L1 (Lasso)
    l2_term = lambda_total * (1 - l1_ratio) * theta[1:]  # L2 (Ridge)

    # Update weights
    theta[1:] -= alpha * (gradient[1:] + l1_term + l2_term)
    theta[0] -= alpha * gradient[0]  # Bias is updated without regularization

# Predictions on test set
y_pred_test = X_test_bias @ theta

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_test - y_pred_test) ** 2)

# Calculate R² Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred_test) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
