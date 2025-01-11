
# Multivariable Linear Regression in Machine Learning

## Overview

Multivariable Linear Regression is an extension of simple linear regression that uses multiple independent variables to predict a dependent variable. The general equation for multivariable linear regression is:

     y = m1*x1 + m2*x2 + m3*x3 + c

Where:

- y is the dependent variable (target variable).
- x1, x2, x3 are the independent variables (features).
- m1, m2, m3 are the coefficients (weights) of the respective features.
- c is the intercept.

The goal is to find the optimal values for m1, m2, m3, and c that minimize the error between the predicted and actual values.

## Required Libraries

To implement a multivariable linear regression model in Python, we need the following libraries:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
```

### Libraries Description:
- **numpy**: A library for numerical computations, primarily used for array manipulation.
- **pandas**: A powerful library for data manipulation and analysis. It is used for loading and preprocessing the dataset.
- **sklearn.linear_model**: Contains the LinearRegression class, which provides the functionality for creating and fitting linear regression models.

## Dataset

The dataset is usually in CSV format, and we can load it using pandas. For example:

```python
data = pd.read_csv('/path/to/dataset.csv')
```

This loads the data into a pandas DataFrame, where you can view and manipulate it.

## Model Creation

To create a linear regression model, instantiate the LinearRegression class:

```python
model = LinearRegression()
```

This object will later be used to fit the model and make predictions.

## Data Preparation

The dataset must be split into features (X) and target (Y). In multivariable regression, the features are the independent variables, and the target is the dependent variable.

```python
X = data[['videos', 'days', 'subscribers']]  # Independent variables
Y = data['views']  # Target variable
```

Here, X contains the columns videos, days, and subscribers, while Y contains the target variable views.

## Training the Model

To train the model, we use the `fit()` method, which learns the relationship between the features and the target variable:

```python
model.fit(X, Y)
```

This step calculates the coefficients $m_1, m_2, m_3$, and intercept $c$ of the model.

## Making Predictions

Once the model is trained, predictions can be made using the `predict()` method. The input for this function should be a 2D array (list of lists) representing the feature values for which we want to predict the target.

```python
predicted_value = model.predict([[45, 180, 3100]])
```

This will output the predicted value based on the given feature values for videos, days, and subscribers.

## Model Coefficients and Intercept

The `coef_` attribute of the model provides the coefficients for each feature. The `intercept_` attribute gives the intercept $c$.

```python
model.coef_  # Coefficients (m1, m2, m3)
model.intercept_  # Intercept (c)
```

Example output:

```python
model.coef_  # [381.732, 42.120, 0.354]
model.intercept_  # 15626.37
```

Coefficients:
- m1 = 381.732
- m2 = 42.120
- m3 = 0.354
- c = 15626.37

## Solving the Equation Manually

To predict a value manually, you can use the following equation:

                   y = m1*x1 + m2*x2 + m3*x3 + c

For example:

```python
y_new = model.coef_[0] * 45 + model.coef_[1] * 180 + model.coef_[2] * 3100 + model.intercept_
```

This manually computes the predicted value based on the coefficients and intercept.

## Summary of Key Methods

### `fit(X, Y)`
**Description**: This method fits the linear regression model on the data by learning the relationships between the features (X) and target (Y).
**Arguments**:
- `X`: A 2D array or DataFrame with the feature values.
- `Y`: A 1D array or Series with the target values.

### `predict(X)`
**Description**: This method predicts the target values based on the learned model using the given feature values.
**Arguments**:
- `X`: A 2D array or list of lists containing the feature values for which predictions are required.

### `coef_`
**Description**: Returns the coefficients of the model, representing the weight of each feature in the regression equation.

### `intercept_`
**Description**: Returns the intercept of the model, which is the constant term $c$ in the equation.
