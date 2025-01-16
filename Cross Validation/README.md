# Cross Validation in Machine Learning

Cross-validation is a statistical technique used to evaluate the performance of a machine learning model. It divides the dataset into subsets to train and test the model multiple times, ensuring reliable and unbiased evaluation.

## Why Use Cross-Validation?

- **Improves Model Evaluation**: Cross-validation provides a more reliable estimate of a model’s performance compared to a single train-test split.
- **Reduces Overfitting**: By testing the model on multiple splits, cross-validation helps identify overfitting or underfitting.
- **Generalizes Better**: It ensures the model works well on unseen data by testing it on different portions of the dataset.



## What is Cross-Validation?

Cross-validation is a method to evaluate the performance of a machine learning model using different subsets of the dataset. It ensures that the model performs consistently across multiple partitions of the data.

### Example:

Using k-fold cross-validation with 5 folds:

1. Split the dataset into 5 equal parts.
2. Use 4 parts for training and 1 part for testing.
3. Repeat the process 5 times, rotating the test fold each time.
4. Calculate the average accuracy from the 5 results.


## How Does Cross Validation Work? (Real World Example)

### Steps:

1. **Split Data**: Divide the dataset into k folds (typically k=5 or k=10).
2. **Train and Test**: For each fold:
    - Train the model on k-1 folds.
    - Test the model on the remaining fold.
3. **Evaluate**: Calculate the model's performance (accuracy, etc.) on each test fold.
4. **Average the Results**: The final performance metric is the average of the scores from each fold.


### Example in Code
Here's an example using Python and the `scikit-learn` library:

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load data
data = pd.read_csv("/DataSets/Iris.csv")
data = data.drop("Id", axis=1)

# Split into features and target
X = data.drop("Species", axis=1)
y = data["Species"]

# Create the model
knn = KNeighborsClassifier()

# Apply Cross-Validation with 5 folds and calculate the mean accuracy
cv_score = cross_val_score(knn, X, y, cv=5)
print(cv_score.mean())
```

### Problem and Solution Approach
In a typical train-test split, you train the model on a training set and test it on a test set. This can sometimes give misleading results if the model is not generalizable. Cross-validation addresses this by evaluating the model on different subsets of data, providing a more accurate performance estimate.

#### Problem: Random Accuracy
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the model and check accuracy (Random approach)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

#### Solution: Cross-Validation
```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Apply Cross-Validation with 5 folds
cv_score = cross_val_score(knn, X, y, cv=5)
print(f"Cross-Validation Mean Accuracy: {cv_score.mean()}")
```

### Cross-Validation with Different Models
Here are examples of applying cross-validation to various machine learning models:

#### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Apply cross-validation
cv_score = cross_val_score(knn, X, y, cv=5)
print(f"KNN Mean Accuracy: {cv_score.mean()}")
```

#### Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Initialize Random Forest Classifier
rf = RandomForestClassifier()

# Apply cross-validation
cv_score = cross_val_score(rf, X, y, cv=5)
print(f"Random Forest Mean Accuracy: {cv_score.mean()}")
```

#### Naive Bayes Classifier
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Initialize Naive Bayes Classifier
nb = GaussianNB()

# Apply cross-validation
cv_score = cross_val_score(nb, X, y, cv=5)
print(f"Naive Bayes Mean Accuracy: {cv_score.mean()}")
```

#### Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Initialize Support Vector Machine
svm = SVC()

# Apply cross-validation
cv_score = cross_val_score(svm, X, y, cv=5)
print(f"SVM Mean Accuracy: {cv_score.mean()}")
```

### Special Methods in the Code

#### `cross_val_score`
This function splits the data into k folds, trains the model on k-1 folds, and tests on the remaining fold. It returns an array of accuracy scores for each fold.

**Arguments:**
- `model`: The machine learning model to evaluate (e.g., `KNeighborsClassifier`, `RandomForestClassifier`).
- `x`: The features of the dataset.
- `y`: The target labels of the dataset.
- `cv`: Number of folds for cross-validation (e.g., `cv=5`).

**Returns:** A list of accuracy scores for each fold.

#### `mean()` method
After running cross-validation, the `mean()` function is used to calculate the average accuracy score from all the folds, giving us a more reliable performance estimate.



### Conclusion
Cross-validation is a powerful technique to evaluate machine learning models by splitting the data into multiple folds. It helps to estimate how well the model performs on unseen data and reduces the chances of overfitting. In the code examples above, we applied cross-validation using different classifiers (KNN, Random Forest, Naive Bayes, and SVM) to get a more accurate measure of their performance.

### 1. cross_val_score

This function from `sklearn.model_selection` is used to evaluate the model using cross-validation. It splits the data into k subsets (called folds), then trains and tests the model on different combinations of these folds. The parameter `cv=5` indicates 5-fold cross-validation.

### 2. Model Initialization

Each machine learning model (KNN, Random Forest, Naive Bayes, SVM) is initialized using a constructor. For example, `knn = KNeighborsClassifier()` initializes the K-Nearest Neighbors model.

### 3. Mean Accuracy Calculation

`cross_val_score` returns an array of accuracy scores from each fold. The `.mean()` method is used to compute the average accuracy across all folds.
