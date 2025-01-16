# Support Vector Machine (SVM) in Machine Learning

Support Vector Machine (SVM) is a supervised machine learning algorithm that is primarily used for classification tasks but can also be applied to regression problems. It is known for its ability to find the optimal hyperplane that separates classes in the feature space.

## Key Concepts

### 1. What is SVM?

SVM is a classification algorithm that works by finding the hyperplane that best separates data points of different classes. The optimal hyperplane maximizes the margin between the nearest data points of each class, which are called support vectors.

### 2. What is a Hyperplane?

A hyperplane is a decision boundary that separates different classes in the feature space. In a two-dimensional space, it is a line, and in higher dimensions, it becomes a plane or a hyperplane.

### 3. How to Find the Optimal Hyperplane?

SVM uses the concept of maximum margin, which is the largest distance between the hyperplane and the nearest data points of each class. These nearest points are the support vectors. The optimization problem is solved to maximize the margin while minimizing classification errors.

In some cases, when data is not linearly separable, SVM uses the kernel trick to map data into a higher-dimensional space where it can find a separating hyperplane.

### 4. Maximum Margin

The margin is the distance between the hyperplane and the nearest data points from each class. SVM maximizes this margin to ensure that the classification is as distinct as possible.

### 5. Support Vectors

Support vectors are the data points that are closest to the hyperplane. These points influence the position and orientation of the hyperplane.

### 6. Non-linear Data and Kernel Trick

When data is not linearly separable, SVM uses a kernel function to transform the data into a higher-dimensional space where a hyperplane can be used to separate the classes. Common kernel functions include:

- Linear Kernel
- Polynomial Kernel
- Radial Basis Function (RBF) Kernel

### Example Transformation

For a dataset where , SVM can map data into a higher-dimensional space using such functions to make it linearly separable.


## Advantages of SVM

- **Versatility**: Can be used for both regression and classification problems.
- **Memory Efficiency**: Stores only a subset of training data (support vectors).
- **Performance**: Works well on small datasets and performs effectively for high-dimensional data.

## Disadvantages of SVM

- **Scalability**: Not well-suited for very large datasets due to high computational cost.
- **Tuning Complexity**: Requires careful parameter tuning (e.g., kernel choice, regularization parameter).

## Applications of SVM

- Face detection
- Text categorization
- Image classification
- Handwriting recognition

## Implementation in Python

### 1. Importing Libraries

```python
import numpy as np
import pandas as pd
```

### 2. Importing Dataset

```python
data = pd.read_csv('/content/drive/MyDrive/DataSets/Iris.csv')
data.head()  # Displays the first five rows
data.sample(5)  # Displays five random rows
```

### 3. Preprocessing the Dataset

#### Removing the "Id" Column

```python
data = data.drop('Id', axis=1)  # Drop the Id column
data.head()
```

#### Data Information Summary

```python
data.info()
```

#### Checking Class Balance

```python
data['Species'].value_counts()
```

### 4. Visualization

#### Pairplot Using Seaborn

```python
import seaborn as sns
sns.pairplot(data, hue='Species')  # Hue parameter represents the target column
```

### 5. Splitting Data into Training and Test Sets

#### Defining Features (X) and Target (Y)

**X-axis:**

```python
x = data.drop('Species', axis=1)  # Drop the target column
x.sample(5)
```

**Y-axis:**

```python
y = data['Species']  # Target column
y.sample(5)
```

#### Splitting the Data

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

### 6. Training the Model

#### Initializing and Fitting the Model

```python
from sklearn.svm import SVC
model = SVC(kernel='poly', C=10)  # Default kernel is 'rbf'
model.fit(x_train, y_train)
```

#### Checking the Kernel Used

```python
model.kernel
```

### 7. Making Predictions

```python
pred = model.predict(x_test)
pred  # Predicted values
y_test  # Actual values
```

### 8. Evaluating the Model

#### Accuracy Score

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)
```

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
```

## Explanation of Special Methods

### data.drop()
Removes a specified column or row from the DataFrame.

**Syntax:** `data.drop('column_name', axis=1)`

### train_test_split()
Splits the dataset into training and testing subsets.

**Parameters:**

- `test_size`: Proportion of the dataset for testing.

### SVC
A class in the `sklearn.svm` module used to initialize the Support Vector Classifier. Parameters include:

- `kernel`: Specifies the kernel type (e.g., 'linear', 'poly', 'rbf').
- `C`: Regularization parameter that balances maximizing the margin and minimizing classification error.

### fit
Fits the model to the training data (`x_train` and `y_train`).

### predict
Predicts the labels for the test data (`x_test`).

### accuracy_score
Calculates the percentage of correctly predicted labels.

### confusion_matrix
Evaluates the performance of the classification model by providing a matrix of true positives, false positives, true negatives, and false negatives.
