
# K-Nearest Neighbors Algorithm

## What is K-Nearest Neighbors Algorithm?

K-Nearest Neighbors (KNN) is a simple, non-parametric, and instance-based machine learning algorithm. It is primarily used for classification and regression problems. The KNN algorithm works by finding the K closest data points to a given input and assigning the majority label (for classification) or averaging the values (for regression).

## How to Choose the Value of K?

1. **Understanding K**: The value of **K** determines how many neighbors are considered when making predictions. A smaller K can make the algorithm sensitive to noise (overfitting), while a larger K makes the algorithm less sensitive to noise but can lead to underfitting.

   - **Increase in K**: As K increases, the model becomes less sensitive to noise but may become too general and underfit the data.
   - **Decrease in K**: A smaller K makes the model more sensitive to noise, which can lead to overfitting.

2. **Example of Choosing K**: In a dataset with 100 samples, choosing K = 3 means the algorithm will consider the 3 closest data points. If K = 50, the algorithm will consider the 50 closest points.

## Finding a Good K Value

The optimal K value is usually found by testing multiple K values and evaluating the model's performance using a validation set or cross-validation. Common approaches include:

- **Cross-validation**: Split the data into subsets and test the model on different subsets.
- **Plotting Error Rates**: Plot the training and validation error for different K values and look for the K that minimizes the error.

## Measuring Distance Between Two Data Points

Distance metrics are important in KNN as they determine how "close" two points are. The most commonly used distance metric is **Euclidean Distance**.

### Euclidean Distance Formula

The Euclidean distance between two points \(P = (x_1, y_1)\) and \(Q = (x_2, y_2)\) in a 2D space is calculated using the formula:

\[
d(P, Q) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]

For higher dimensions, the formula generalizes as:

\[
d(P, Q) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + \dots + (z_1 - z_2)^2}
\]

Where \(x_1, x_2, y_1, y_2, \dots, z_1, z_2\) are the corresponding coordinates of the two data points in each dimension.

## The Iris Dataset

### Overview

The Iris dataset is a well-known dataset for classification. It contains 150 samples of iris flowers, with 4 features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Each sample belongs to one of three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## K-Nearest Neighbors Algorithm on the Iris Dataset

### 1. Import Required Libraries
```python
import numpy as np
import pandas as pd
```
We import numpy for numerical computations and pandas for data manipulation.

### 2. Load the Dataset
```python
data = pd.read_csv('/content/drive/MyDrive/DataSets/Iris.csv')
data.head()
```
We load the Iris dataset into a pandas DataFrame and display the first 5 rows.

### 3. Check Last 5 Rows of Data
```python
data.tail()
```
Displays the last 5 rows of the dataset.

### 4. Dataset Information
```python
data.info()
```
Displays the information about the dataset, including column names and data types.

### 5. Feature and Target Variables
```python
x = data.iloc[:,1:5]
y = data.iloc[:,-1]
```
- `x`: Features (sepal length, sepal width, petal length, petal width).
- `y`: Target (species).

## 6. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
```
We scale the features to normalize the data (important for K-NN, as it is sensitive to feature scaling).

### 7. Splitting the Dataset into Training and Test Sets
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
We split the dataset into training and testing sets (80% for training, 20% for testing).

### 8. Training the Model Using K-NN
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
```
We create and train a K-NN model using `n_neighbors=1`.

### 9. Making Predictions
```python
pred = model.predict(x_test)
```


## 10. Model Accuracy
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)
```
We calculate the accuracy of the model by comparing the predicted species with the actual species in the test set.

## 11. Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
```
The confusion matrix shows the count of true positives, false positives, true negatives, and false negatives.

## 12. Choosing the Optimal K Value
```python
correct_sum = []
for i in range(1,21):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    correct = np.sum(pred == y_test)
    correct_sum.append(correct)
```
We evaluate the model for different values of K (from 1 to 20) and store the number of correct predictions for each value of K.

## 13. Final Accuracy and Confusion Matrix for K=8
```python
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy_score(y_test, pred)
```
We train and test the model using `K=8`, then check its accuracy and confusion matrix.

It looks like you want to convert the provided content into a Markdown file. Here is the content formatted in Markdown:


## Special Methods in Code

### 1. StandardScaler

Scales data to have zero mean and unit variance.

**Example:**

```python
scaler = StandardScaler()
x = scaler.fit_transform(x)
```

### 2. train_test_split

Splits the dataset into training and testing sets.

**Example:**

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

### 3. KNeighborsClassifier

Implements the KNN algorithm.

**Parameters:**

- `n_neighbors`: Number of neighbors (K).

**Example:**

```python
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)
```

### 4. accuracy_score

Computes the accuracy of predictions.

**Example:**

```python
accuracy = accuracy_score(y_test, pred)
```

###5. confusion_matrix

Generates a matrix to evaluate model performance.

**Example:**

```python
cm = confusion_matrix(y_test, pred)
```

## Summary

The KNN algorithm is simple and effective for classification tasks.

Choosing the optimal K value is crucial to balancing underfitting and overfitting.

The Iris dataset is a great starting point for practicing KNN.


## Conclusion
- The K-Nearest Neighbors algorithm is a simple and effective classification technique.
- Feature scaling is essential, as K-NN is sensitive to the magnitude of features.
- The optimal value of K should be chosen carefully to avoid overfitting or underfitting.
