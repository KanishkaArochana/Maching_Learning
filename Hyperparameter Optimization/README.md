# Hyperparameter Optimization

Hyperparameter optimization refers to the process of finding the best set of hyperparameters for a machine learning model to maximize its performance. Hyperparameters are the parameters that are not learned directly from the training data but are set before the training process begins. Examples of hyperparameters include the learning rate, number of trees in a random forest, and parameters specific to certain algorithms like C and kernel in Support Vector Machines (SVMs).

### Parameters in SVM

**Kernel:**

The kernel determines the function used to transform the data into a higher-dimensional space to make it linearly separable.

Common kernel types:

- **linear**: Uses a linear hyperplane to separate data.
- **rbf (Radial Basis Function)**: Suitable for non-linear problems.
- **poly (Polynomial)**: Adds polynomial features for non-linear classification.

**C:**

The C parameter controls the trade-off between achieving a low error on the training data and maintaining a simple decision boundary (generalization).

- A smaller value of C leads to a simpler model, while a larger C focuses on fitting the training data better.

### Methods for Hyperparameter Optimization

There are two popular methods for hyperparameter optimization in machine learning:

## 1. GridSearchCV

GridSearchCV performs an exhaustive search over a specified parameter grid by trying all possible combinations of the parameters. It evaluates each combination using cross-validation and selects the one with the best performance.

**Example:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf', 'linear', 'poly']}

# Create an instance of the SVM model
model = SVC()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid)
grid_search.fit(x_train, y_train)

# Output the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", grid_search.score(x_test, y_test))
```

**Output:**

```
Best Parameters: { 'C': 10, 'kernel': 'rbf' }
Test Accuracy: 0.95 (example value)
```

**Explanation:**

- The `param_grid` specifies the range of values for `C` and `kernel`.
- `GridSearchCV` evaluates all combinations of these hyperparameters.
- `fit(x_train, y_train)`: Trains the model for all parameter combinations.
- `best_params_`: Retrieves the combination of parameters that achieved the best results.
- `score(x_test, y_test)`: Evaluates the model with the best parameters on the test set.

**Advantages of GridSearchCV:**

-Exhaustive search ensures the best hyperparameter combination within the defined grid.

**Disadvantages:**

-Computationally expensive, especially when the grid size is large.

## 2. RandomizedSearchCV

RandomizedSearchCV performs a random search over the parameter grid, selecting a fixed number of random combinations to evaluate. This is faster than GridSearchCV, especially when the parameter space is large.

**Example:**

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

# Define the parameter distribution
param_dist = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf', 'linear', 'poly']}

# Create an instance of the SVM model
model = SVC()

# Perform RandomizedSearchCV
randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5)
randomized_search.fit(x_train, y_train)

# Output the best parameters and score
print("Best Parameters:", randomized_search.best_params_)
print("Test Accuracy:", randomized_search.score(x_test, y_test))
```

**Output:**

```
Best Parameters: { 'C': 1, 'kernel': 'linear' }
Test Accuracy: 0.92 (example value)
```
**Explanation:**
- param_distributions: Defines the range of values for `C` and `kernel`.
- `n_iter`: Specifies the number of parameter combinations to evaluate.
- `fit(x_train, y_train)`: Trains the model for all parameter combinations.
- `best_params_`: Retrieves the combination of parameters that achieved the best results.
- `score(x_test, y_test)`: Evaluates the model with the best parameters on the test set.**

**Advantages of RandomizedSearchCV**

- Faster than GridSearchCV because it evaluates only a random subset of combinations.
- Suitable for large hyperparameter spaces.

**Disadvantages**
- May miss the optimal combination since it does not evaluate all possibilities.

### Differences Between GridSearchCV and RandomizedSearchCV

| Feature          | GridSearchCV                              | RandomizedSearchCV                        |
|------------------|-------------------------------------------|-------------------------------------------|
| **Search Method**| Exhaustive search of all combinations.    | Randomly selects combinations to evaluate.|
| **Execution Time**| Longer, especially for large parameter spaces.| Faster, as fewer combinations are evaluated.|
| **Flexibility**  | Covers all possible combinations.         | Useful for exploring large spaces efficiently.|

### Implementation

**Dataset Preparation**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/DataSets/Iris.csv')

# Split features and target
x = data.drop(['Species', 'Id'], axis=1)
y = data['Species']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
```

**Model Training without Optimization**

```python
from sklearn.svm import SVC

# Create and train the SVM model
model = SVC(C=1, kernel='rbf')
model.fit(x_train, y_train)

# Evaluate the model
print("Test Accuracy:", model.score(x_test, y_test))
```

**GridSearchCV Implementation**

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf', 'linear', 'poly']}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid)
grid_search.fit(x_train, y_train)

# Output the results
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", grid_search.score(x_test, y_test))
```

**RandomizedSearchCV Implementation**

```python
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter distribution
param_dist = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf', 'linear', 'poly']}

# Perform RandomizedSearchCV
randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5)
randomized_search.fit(x_train, y_train)

# Output the results
print("Best Parameters:", randomized_search.best_params_)
print("Test Accuracy:", randomized_search.score(x_test, y_test))
```

## Special Methods in the Code

### 1. GridSearchCV.best_params_

**Purpose:** Returns the best combination of hyperparameters found during the grid search.

**Usage:**

```python
print(grid_search.best_params_)
```

### 2. GridSearchCV.score()

**Purpose:** Evaluates the performance of the model with the best parameters on the test data.

**Usage:**

```python
print(grid_search.score(x_test, y_test))
```

### 3. RandomizedSearchCV.best_params_

**Purpose:** Returns the best combination of hyperparameters found during the randomized search.

**Usage:**

```python
print(randomized_search.best_params_)
```

### 4. RandomizedSearchCV.score()

**Purpose:** Evaluates the performance of the model with the best parameters on the test data.

**Usage:**

```python
print(randomized_search.score(x_test, y_test))
```


This document provides a clear overview of hyperparameter optimization using GridSearchCV and RandomizedSearchCV, with examples demonstrating their usage in Python for an SVM model. The differences between the two methods are highlighted to help decide which approach suits a specific problem.

