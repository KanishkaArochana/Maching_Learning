# Logistic Regression in Machine Learning

## Overview

Logistic regression is a statistical method used for solving classification problems. It predicts the probability that a given input belongs to a particular category. The output of logistic regression is always between 0 and 1, making it ideal for binary and multi-class classification.

## Types of Classification Problems

### Binary Classification

Predicts one of two categories.

**Example:** Classifying images as either a dog or a cat.

### Multi-Class Classification

Predicts one of three or more categories.

**Example:** Classifying fruits as apples, bananas, or oranges.

## Why Use Logistic Regression for Classification Problems?

Logistic regression is preferred for classification because:

- It models the relationship between input features and the probability of a particular class.
- It uses the sigmoid function, ensuring predictions are probabilities.

## Logistic Regression Formula

The logistic regression model predicts the probability of an output using the equation:

$$ P(y=1|x) = \frac{1}{1 + e^{-(b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n)}} $$

- \( P(y=1|x) \): Probability of the positive class.
- \( b_0 \): Intercept.
- \( b_1, b_2, ..., b_n \): Coefficients for the input features.
- \( e \): Euler’s number (approximately 2.718).

The sigmoid function maps any real number to a value between 0 and 1.

## Implementation of Logistic Regression Using Python

### Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

These libraries are used for:

- Numerical operations (NumPy).
- Data manipulation (Pandas).
- Visualization (Matplotlib).

### Import Dataset

```python
data = pd.read_csv('/content/drive/MyDrive/DataSets/Logistic Regression Dataset.csv')
```

Here, the dataset is imported into a DataFrame for analysis.

### Display Dataset

Display the entire dataset using:

```python
data
```

Display the first five rows:

```python
data.head()
```

Display the last five rows:

```python
data.tail()
```

### Visualize Data

Use a scatter plot to understand the relationship between variables:

```python
plt.scatter(data.age, data.job)
plt.xlabel("Age")
plt.ylabel("Job")
plt.title("Age vs Job")
plt.show()
```

### Define X and Y Values

X (Independent variable) should be a 2D array:

```python
x = data[['age']]
x
```

Y (Dependent variable) should not be a 2D array:

```python
y = data['job']
y
```

### Train and Test Values

Split the dataset into training and testing sets:

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

- `x_train`: Training data for input features.
- `x_test`: Testing data for input features.
- `y_train`: Training data for target values.
- `y_test`: Testing data for target values.

Check the number of training and testing samples:

```python
len(x_train)
len(x_test)
```

### Build the Model

Import the logistic regression class:

```python
from sklearn.linear_model import LogisticRegression
```

Create a logistic regression object:

```python
model = LogisticRegression()
```

Train the model using the `fit()` method:

```python
model.fit(x_train, y_train)
```

This method trains the model on the provided training data.

### Predict Values

Predict the target values for the test set:

```python
predictions = model.predict(x_test)
```

Compare predictions with actual values:

```python
x_test
y_test
predictions
```

### Evaluate Accuracy

Measure the accuracy of the model using the `score()` method:

```python
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")
```

`score()` compares predicted values with actual values and returns the accuracy.

### Test the Model with New Data

Predict the output for new values:

```python
new_age = [[24]]
print(model.predict(new_age))

new_age = [[29]]
print(model.predict(new_age))

ages = np.array([[31], [22], [34]])
print(model.predict(ages))
```

## Summary of Special Methods

- `train_test_split()`: Splits the dataset into training and testing sets.
- `LogisticRegression()`: Creates a logistic regression model.
- `fit()`: Trains the logistic regression model on the training data.
- `predict()`: Predicts the target values for given inputs.
- `score()`: Evaluates the accuracy of the model.
## Logistic Regression Plot

Below is an illustration of the logistic regression plot showing the sigmoid function:

!Logistic Regression Sigmoid Plot

### Description of the Plot:

- **X-axis**: Feature values (e.g.,).
- **Y-axis**: Probability values ().
- **Curve**: Sigmoid function, which maps any real number to a range between 0 and 1.
- **Threshold**: A horizontal line at  used to classify data points.

### Differences Between Linear Regression and Logistic Regression

| Aspect       | Linear Regression                  | Logistic Regression                  |
|--------------|------------------------------------|--------------------------------------|
| **Purpose**  | Predicts continuous numerical values. | Predicts categorical class labels.   |
| **Output**   | Continuous range (e.g., -∞ to ∞).  | Probability values (0 to 1).         |
| **Model**    | Fits a straight line.              | Fits an "S-shaped" logistic curve.   |
| **Use Case** | Regression problems.               | Classification problems.             |


