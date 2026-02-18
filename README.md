# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Required Libraries
2.Load and Inspect the Dataset
3.Select Multiple Input Features and Target Variable
4.Split the Dataset into Training and Testing Sets
5.Perform Feature Scaling on Input Variables
6.Initialize and Configure the SGD Regressor Model
7.Train the Model Using Training Data
8.Predict Output for Test Data
9.Evaluate Model Performance Using Error Metrics 

## Program:
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# Load the California Housing dataset
data = fetch_california_housing()
# Use the first 3 features as inputs
X = data.data[:, :3] # Features: 'MedInc', 'HouseAge', 'AveRooms'
# Use 'MedHouseVal' and 'AveOccup' as output variables
Y = np.column_stack((data.target, data.data[:, 6])) # Targets: 'MedHouseVal', 'AveOccup'
22
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)
# Train the model
multi_output_sgd.fit(X_train, Y_train)
# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)
# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
# Optionally, print some predictions
print("\nPredictions:\n", Y_pred[:5]) # Print first 5 predictions
```
```
Developed by: DWIJESH RAJ SINHA Y
RegisterNumber: 25013468
```

## Output:
<img width="910" height="617" alt="image" src="https://github.com/user-attachments/assets/094e4556-c649-4bc8-84e1-4af690a842bf" />
<img width="865" height="153" alt="image" src="https://github.com/user-attachments/assets/1e6a997e-6edb-4369-b2cc-6a1d7e6c8f89" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
