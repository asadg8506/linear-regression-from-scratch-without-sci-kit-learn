Linear Regression using NumPy and Pandas

Overview
This project demonstrates how to implement Linear Regression from scratch using only NumPy and Pandas, without relying on libraries like scikit-learn. The implementation covers:
- Data preprocessing using Pandas
- Computing the optimal parameters using the Normal Equation
- Gradient Descent optimization
- Model evaluation using Mean Squared Error (MSE)

Features
- Simple Linear Regression (one independent variable)
- Multiple Linear Regression (multiple independent variables)
- Normalization of input features
- Model training using Gradient Descent and Normal Equation
- Performance evaluation with MSE

Installation
Ensure you have Python installed along with NumPy and Pandas. You can install dependencies using:

```bash
pip install numpy pandas
```

Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/asadg8506/linear-regression-numpy-pandas.git
   cd linear-regression-numpy-pandas
   ```
2. Run the main script:
   ```bash
   python linear_regression.py
   ```
3. Modify the dataset or parameters as needed in the script.

File Structure
```
├── data.csv                    # Sample dataset
├── linear_regression.py         # Implementation of Linear Regression
├── README.md                    # Project documentation
```

Implementation Details
1. Data Loading & Preprocessing
- Data is loaded using Pandas.
- Missing values are handled appropriately.
- Features are normalized for better performance in Gradient Descent.

2. Model Training
Two approaches are implemented:
- **Normal Equation**: Closed-form solution for linear regression.
- **Gradient Descent**: Iterative approach to minimize cost function.

3. Model Evaluation
- Mean Squared Error (MSE) is used to measure model performance.
- Predictions can be compared against actual values.

📁 Dataset
https://www.kaggle.com/datasets/kyanyoga/sample-sales-data

Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

License
This project is licensed under the MIT License.

