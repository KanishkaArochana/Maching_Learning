# Random Forest Algorithm in Machine Learning

## What is Random Forest Algorithm?

Random Forest is an ensemble learning algorithm used for classification and regression tasks. It builds multiple decision trees and combines their predictions to improve accuracy and avoid overfitting. Each decision tree in the forest is trained on a random subset of the training data, and each decision is made by aggregating the predictions of all the trees.

**Key Features:**
- It is used for both classification and regression problems.
- It is a supervised learning algorithm.
- It uses multiple decision trees to make predictions.

### Number of Decision Trees in Random Forest:
The number of decision trees in a Random Forest is a hyperparameter that can be set by the user. The default value is typically 100, but it can be adjusted based on the problem and computational resources. In this example, we will use 50 decision trees.

## How Does Random Forest Work?

1. **Random Sampling:** The data is divided into random subsets.
2. **Tree Building:** A decision tree is trained on each subset.
3. **Prediction:** Each tree makes a prediction, and the final prediction is determined by majority voting (for classification) or averaging (for regression).
4. **Final Prediction:** The most frequent prediction across all trees is taken as the final prediction.

Random Forest typically provides better accuracy than a single decision tree because it reduces overfitting and improves generalization.

## Advantages of Random Forest:
- High accuracy.
- Handles both classification and regression problems.
- Reduces overfitting compared to a single decision tree.
- Can handle missing values and large datasets.
- Handles non-linear data relationships well.

## Disadvantages of Random Forest:
- Can be computationally expensive.
- Less interpretable compared to a single decision tree.
- May take longer to train for large datasets.

## Difference Between Decision Tree and Random Forest:

| Aspect                     | Decision Tree                        | Random Forest                    |
|----------------------------|--------------------------------------|----------------------------------|
| **Number of Trees**         | One tree                            | Multiple trees (ensemble)         |
| **Accuracy**                | Lower (prone to overfitting)         | Higher (due to aggregation)       |
| **Overfitting**             | More prone to overfitting           | Less prone to overfitting         |
| **Model Complexity**        | Simple to interpret and visualize   | Harder to interpret and visualize |
| **Training Time**           | Faster                              | Slower due to multiple trees      |

---

## Code Walkthrough

### Importing Libraries

```python
import numpy as np
import pandas as pd
```
Data Handling with NumPy and Pandas

## Load Dataset
```python
import pandas as pd

data = pd.read_csv('/content/drive/MyDrive/DataSets/kyphosis.csv')
```
This loads the dataset from the given path into a Pandas DataFrame.

## Dataset Overview
```python
data.head()  # Show first 5 rows
data.tail()  # Show last 5 rows
data.shape  # Show size of data
data.info()  # Show dataset information
```
- `head()`: Displays the first 5 rows of the dataset.
- `tail()`: Displays the last 5 rows of the dataset.
- `shape`: Shows the number of rows and columns in the dataset.
- `info()`: Provides information about the dataset, such as column types and missing values.

## Feature and Target Variables
```python
x = data.drop('Kyphosis', axis=1)  # Remove Kyphosis column as feature variables
y = data['Kyphosis']  # Target variable
```
- `x`: Feature variables (everything except Kyphosis).
- `y`: Target variable (Kyphosis).

## Splitting Data into Train and Test Sets
```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # 30% for testing
```
- `train_test_split`: Splits the dataset into training and testing sets.
- `test_size=0.3`: 30% of the data will be used for testing, and the rest will be used for training.

## Decision Tree Model
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)  # Train the model
```
- `DecisionTreeClassifier()`: Initializes a decision tree classifier.
- `fit()`: Trains the model using the training data.

## Predictions with Decision Tree
```python
pred = model.predict(x_test)
```
- `predict()`: Makes predictions on the test data.

## Accuracy and Evaluation of Decision Tree
```python
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test, pred)  # Accuracy score
confusion_matrix(y_test, pred)  # Confusion matrix
```
- `accuracy_score()`: Calculates the accuracy of the model by comparing predictions to actual values.
- `confusion_matrix()`: Generates a confusion matrix to show the performance of the model.

## Random Forest Model
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50)  # 50 decision trees
model.fit(x_train, y_train)  # Train the model
```
- `RandomForestClassifier()`: Initializes the Random Forest classifier.
- `n_estimators=50`: Specifies that the model will use 50 decision trees.
- `fit()`: Trains the Random Forest model.

## Predictions with Random Forest
```python
pred = model.predict(x_test)
```
- `predict()`: Makes predictions using the trained Random Forest model.

## Accuracy and Evaluation of Random Forest
```python
accuracy_score(y_test, pred)  # Accuracy score
confusion_matrix(y_test, pred)  # Confusion matrix
```
- `accuracy_score()`: Calculates the accuracy of the Random Forest model.
- `confusion_matrix()`: Generates a confusion matrix for the Random Forest model.

## Conclusion
In this example, we compared the performance of a Decision Tree and a Random Forest on the kyphosis dataset. As shown, Random Forest generally provides better accuracy due to the aggregation of multiple decision trees, reducing the risk of overfitting.
