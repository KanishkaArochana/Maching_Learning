
# Linear Regression in Machine Learning

## What is Linear Regression?

Linear Regression is one of the simplest and most commonly used algorithms in machine learning. It is a method used to model the relationship between one or more independent variables (input features) and a dependent variable (output) by fitting a linear equation to the data. The main goal is to predict the dependent variable based on the given inputs.

## Why Use Linear Regression in Machine Learning?

Linear Regression is widely used because:

- **Simplicity**: It is easy to understand and implement.
- **Interpretability**: The coefficients of the linear equation can provide insights into the importance and relationship of each feature.
- **Efficiency**: It works well for linearly separable data and requires less computational power.
- **Foundation for Other Algorithms**: It serves as a building block for more complex regression and classification algorithms.

## Linear Regression as a Supervised Learning Algorithm

Linear Regression is a supervised learning algorithm because it learns from labeled data. In this context:

- The input features (X) are the independent variables.
- The output (Y) is the dependent variable.

The algorithm tries to minimize the difference between the actual values (observed) and the predicted values by finding the best-fit line.

## How to Search for the Best-Fit Line

The best-fit line is determined by minimizing the error between predicted and actual values. This is typically achieved using the Least Squares Method, which minimizes the sum of the squared differences between the observed and predicted values:

Where:

- Actual value
- Predicted value

The algorithm adjusts the coefficients (weights) of the line iteratively to reduce this error.

## Single Variable Linear Regression

In single-variable (univariate) linear regression, the relationship is expressed as:

y = mx +c

Terms:

-  Predicted value (output)
-  Independent variable (input)
-  Slope of the line (how much \( y \) changes with \( x \))
-  Intercept (the value of \( y \) when \( x = 0 \))

The slope ( m ) and intercept ( c ) are learned during training.

## Multi-Variable Linear Regression

In multi-variable (multivariate) linear regression, the relationship is extended to include multiple input features:

y = m1*x1 + m2*x2 + m3*x3 + c

Terms:

- Predicted value (output)
-  Independent variables (input features)
- Coefficients (weights) for each feature
-  Intercept

The model learns the weights (  m1*x1 + m2*x2 + m3*x3 ) and intercept ( c ) to best fit the training data.

The formula can also be represented in vectorized form for simplicity:



Where:

- Matrix of input features
- Vector of coefficients (weights)
-  Intercept

## Summary

Linear Regression is a fundamental supervised learning algorithm that provides a simple yet powerful way to predict outcomes based on input features. Its ability to provide interpretable models makes it a preferred choice for solving regression problems in machine learning.
