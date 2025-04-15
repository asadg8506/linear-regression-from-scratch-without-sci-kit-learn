# Linear Regression from Scratch with L1, L2 & ElasticNet (No Scikit-learn)

## Overview

This project demonstrates how to implement Linear Regression from scratch using only NumPy and Pandas, without relying on libraries like scikit-learn. The implementation covers:

- Data preprocessing using Pandas  
- Computing the optimal parameters using the Normal Equation  
- Gradient Descent optimization  
- Model evaluation using Mean Squared Error (MSE)  
- Regularization techniques: L1 (Lasso), L2 (Ridge), and ElasticNet

## Features

- Simple Linear Regression (one independent variable)    
- Normalization of input features  
- Model training using Gradient Descent and Normal Equation  
- L1, L2, and ElasticNet Regularization  
- Performance evaluation with MSE  

## Installation

Ensure you have Python installed along with NumPy and Pandas. You can install dependencies using:

pip install numpy pandas
Usage
Clone the repository:

git clone https://github.com/asadg8506/linear-regression-numpy-pandas.git
cd linear-regression-numpy-pandas
Run the desired script:

python linear_regression.py         # For basic linear regression
python ridge_regression.py          # For L2 regularization
python lasso_regression.py          # For L1 regularization
python elasticnet_regression.py     # For ElasticNet
Modify the dataset or parameters as needed in each script.

File Structure
├── data.csv                     # Sample dataset
├── linear_regression.py         # Basic Linear Regression
├── ridge_regression.py          # Ridge (L2 Regularization)
├── lasso_regression.py          # Lasso (L1 Regularization)
├── elasticnet_regression.py     # ElasticNet (L1 + L2 Regularization)
├── utils.py                     # Helper functions
├── README.md                    # Project documentation
Implementation Details
Data Loading & Preprocessing
Data is loaded using Pandas

Missing values are handled appropriately

Features are normalized for better performance in Gradient Descent

Model Training
Two approaches are implemented:

Normal Equation: Closed-form solution for linear regression

Gradient Descent: Iterative approach to minimize cost function

L1/L2/ElasticNet: Regularized cost functions to prevent overfitting

Model Evaluation
Mean Squared Error (MSE) is used to measure model performance

Predictions can be compared against actual values

📁 Dataset
You can use your own dataset or try the sample one:
https://www.kaggle.com/datasets/kyanyoga/sample-sales-data

Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

License
This project is licensed under the MIT License.
