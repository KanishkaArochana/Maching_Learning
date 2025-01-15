# Naive Bayes Algorithm Documentation
## Bayes' Theorem

Bayes' Theorem is a mathematical formula used to determine the probability of an event based on prior knowledge of conditions that might be related to the event. The formula is:

`P(A | B) = [P(B | A) * P(A)] / P(B)`

Where:

- **P(A|B)**: Probability of event A occurring given B is true (Posterior Probability).
- **P(B|A)**: Probability of event B occurring given A is true (Likelihood).
- **P(A)**: Probability of event A (Prior Probability).
- **P(B)**: Probability of event B (Evidence).

## How Naive Bayes Works

Naive Bayes assumes all features are independent of each other (hence "naive").
For a given set of features, it calculates the posterior probability for each class using Bayes' Theorem.
The algorithm selects the class with the highest probability as the prediction.

### Steps:

1. Calculate the prior probabilities for each class.
2. Compute the likelihood for each feature given the class.
3. Apply Bayes' Theorem to calculate the posterior probability.
4. Choose the class with the highest posterior probability.

## Types of Naive Bayes

### a. Multinomial Naive Bayes
Used for multi-class classification problems. It works well for text classification where data is represented as term frequencies or counts.

Example using sklearn:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

## Sample dataset
data = ["I love this movie", "This movie is bad", "Amazing film", "Horrible movie"]
labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

## Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

## Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

## Train model
model = MultinomialNB()
model.fit(X_train, y_train)

## Predict
predictions = model.predict(X_test)
print("Predictions:", predictions)
```

### b. Bernoulli Naive Bayes
Used for binary classification problems, where features are binary (e.g., presence/absence of a term).

Example:

```python
from sklearn.naive_bayes import BernoulliNB

## Binary dataset
X = [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]]  # Binary features
y = [1, 0, 1, 0]  # Binary labels

## Train model
model = BernoulliNB()
model.fit(X, y)

## Predict
print("Predictions:", model.predict([[1, 0, 0], [0, 1, 1]]))
```

### c. Gaussian Naive Bayes
Used when features are continuous values and follows a Gaussian distribution.

Example:

```python
from sklearn.naive_bayes import GaussianNB

## Continuous dataset
X = [[1.0, 2.1], [1.5, 1.8], [3.0, 3.2], [5.0, 6.8]]
y = [0, 0, 1, 1]  # Binary labels

## Train model
model = GaussianNB()
model.fit(X, y)

## Predict
print("Predictions:", model.predict([[2.0, 2.5], [4.5, 5.0]]))
```

## Advantages and Disadvantages

### Advantages
- Simple and Fast: Efficient for large datasets.
- Performs well in multi-class problems.
- Can handle both binary and multi-class classification.

### Disadvantages
- Assumes all features are independent, which is rarely true in real-world problems.
- Sensitive to the quality of data, especially when feature correlations exist.

## Applications

**Classification Problems:**
- Handwritten digit recognition
- Image classification

**Regression Problems:**
- Predicting house prices
- Stock market trends

**Recommendation Systems:**
- Movie or product recommendations based on user preferences

### a. Spam Filtering
Classifies emails as spam or not spam based on the occurrence of words.

### b. Sentiment Analysis
Predicts the sentiment (positive or negative) of text, such as product reviews.

### c. Recommendation Systems
Used to suggest items like products or movies based on user behavior.

Example (Spam Classification):

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

## Sample emails
emails = ["Win a free lottery", "Hello, how are you?", "Claim your free prize now", "Meeting at 3 PM"]
labels = [1, 0, 1, 0]  # 1: Spam, 0: Not Spam

## Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

## Train model
model = MultinomialNB()
model.fit(X, labels)

## Predict
test_emails = ["Get a free iPhone", "Let's catch up tomorrow"]
test_features = vectorizer.transform(test_emails)
print("Predictions:", model.predict(test_features))
```
## Implementing KNN in Python

## Naive Bayes Algorithm Documentation for Machine Learning

Naive Bayes is a simple but powerful classification algorithm based on Bayes' theorem, which works on the principle of conditional probability. It's widely used for classification tasks, especially when dealing with large datasets.

In this documentation, we will explain the steps for implementing the Naive Bayes algorithm using the GaussianNB model from `sklearn.naive_bayes`. We will break down the special methods and steps used in the code.

## Steps to Implement Naive Bayes Algorithm

### 1. Importing Necessary Libraries

```python
import pandas as pd
import numpy as np
```

- `pandas` is used for data manipulation and analysis.
- `numpy` is used for numerical operations.

### 2. Loading the Dataset

```python
data = pd.read_csv('/content/drive/MyDrive/DataSets/kyphosis.csv')
```

The dataset is loaded from a CSV file using pandas. In this case, we are using the kyphosis dataset.

### 3. Exploring the Dataset

```python
data.head()  # First five data rows
data.tail()  # Last five data rows
data.shape  # Shape of the dataset
data.info()  # Information about the dataset
```

- `head()`: Displays the first 5 rows of the dataset.
- `tail()`: Displays the last 5 rows of the dataset.
- `shape`: Returns the number of rows and columns in the dataset.
- `info()`: Provides summary information about the dataset, such as column names, data types, and missing values.

### 4. Visualization Techniques

```python
import seaborn as sns
sns.pairplot(data, hue="Kyphosis")
sns.countplot(x="Kyphosis", data=data)
```

- `pairplot()`: A visualization that shows pairwise relationships in the dataset, where each pair of columns is plotted against each other. We use the "Kyphosis" column to color the plots.
- `countplot()`: A count plot is used to show the distribution of the target variable "Kyphosis", which helps to identify if the dataset is imbalanced.

### 5. Data Pre-Processing

#### X-axis (Features)

```python
x = data.drop('Kyphosis', axis=1)
```

- `x`: This stores all the feature columns by dropping the target column 'Kyphosis'. `axis=1` is used to specify that we are dropping a column (not a row).

#### Y-axis (Target)

```python
y = data['Kyphosis']
```

- `y`: This stores the target variable (the class label), which is the 'Kyphosis' column in the dataset.

#### Train and Test Split

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
```

- `train_test_split()`: This function splits the dataset into training and testing sets. 70% of the data is used for training (`x_train`, `y_train`) and 30% is used for testing (`x_test`, `y_test`). The `test_size=0.3` parameter specifies that 30% of the data should be reserved for testing.

### 6. Naive Bayes Algorithm

#### Importing the Gaussian Naive Bayes Model

```python
from sklearn.naive_bayes import GaussianNB
```

- `GaussianNB`: This is the Naive Bayes model used for classification tasks. It assumes that the features follow a Gaussian (normal) distribution, which is why it’s suitable for datasets with continuous values.

#### Creating and Training the Model

```python
NB = GaussianNB()  # Create model
NB.fit(x_train, y_train)  # Train model on training data
```

- `GaussianNB()`: This creates an instance of the Gaussian Naive Bayes model.
- `fit()`: The model is trained on the training data (`x_train` and `y_train`). The algorithm learns the relationships between the features and the target variable.

### 7. Prediction

```python
pred = NB.predict(x_test)
```

- `predict()`: This method is used to make predictions on the test data (`x_test`). The model uses the learned relationships to predict the class labels for the test set.

#### Predicted Values and Actual Values

```python
pred  # Predicted values for y
y_test  # Actual values for y
```

- `pred`: This stores the predicted class labels for the test data.
- `y_test`: These are the actual class labels for the test data.

### 8. Evaluation Metrics

#### Accuracy Score

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)
```

- `accuracy_score()`: This method calculates the accuracy of the model by comparing the predicted values (`pred`) with the actual values (`y_test`). Accuracy is the ratio of correct predictions to total predictions.

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
```

- `confusion_matrix()`: This method returns a confusion matrix, which shows how many true positives, false positives, true negatives, and false negatives the model has made. It helps to evaluate the performance of the classification model, especially in cases of imbalanced datasets.

## Explanation of Naive Bayes Algorithm

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that the features are independent, hence "naive". The algorithm computes the probability of each class and assigns the class with the highest probability to each sample.

The formula for Bayes' theorem is:

P(C_k|X) = (P(X|C_k) * P(C_k)) / P(X)


Where:
- **P(C_k|X)** is the posterior probability of class \(C_k\) given the features \(X\).
- **P(X|C_k)** is the likelihood of observing the features \(X\) given class \(C_k\).
- **P(C_k)** is the prior probability of class \(C_k\).
- **P(X)** is the evidence or normalization term.

## Special Methods in the Code

- `GaussianNB()`: The Gaussian Naive Bayes algorithm is specifically used when the features are continuous and assumed to follow a Gaussian distribution. It estimates the parameters (mean and variance) for each feature in each class.
- `fit()`: This method fits the Naive Bayes model to the training data by calculating the required probabilities from the input data.
- `predict()`: After the model is trained, `predict()` is used to generate class predictions for new data based on the learned probabilities.

## Conclusion

Naive Bayes is a fast and effective classification algorithm, especially when dealing with large datasets or when the features are independent. It's particularly useful for text classification, spam detection, and medical diagnosis problems. However, its assumption of feature independence can sometimes be a limitation, especially in cases where features are strongly correlated.
