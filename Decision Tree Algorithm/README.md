# Decision Tree Algorithm in Machine Learning

The Decision Tree Algorithm is a supervised learning method used for solving both regression and classification problems, though it is most commonly applied to classification tasks. It works by splitting the data into subsets based on feature values, creating a tree-like structure of decisions.

## Key Components of Decision Trees

### Nodes

- **Decision Node**: Represents a decision to be made based on a feature.
- **Root Node**: The topmost node in the tree that initiates the splitting process.
- **Leaf/Terminal Node**: Represents the final outcome or prediction.

### Edges and Branches

- **Edges**: Connect nodes in the tree.
- **Branch**: Represents the outcome of a decision and connects one node to another.
- **Parent/Child Nodes**: A parent node splits into child nodes based on the decisions.

## Example of Decision Tree Structure

```css
Root Node
   |
   |--- Branch 1 ---> Child Node 1
   |                    |
   |                    |--- Leaf Node 1
   |
   |--- Branch 2 ---> Child Node 2
                        |
                        |--- Leaf Node 2
```

```css
          [Age]
         /     \
     <=30        >30
     /   \       /   \
  [Income] [Income]  Yes  No
    |      |   
   <50    >50   
    |      |
  No     Yes
```

- The root node is the "Age" feature.
- The leaf nodes represent the decision outcome ("Yes" or "No").
- The edges are the conditions based on the values of "Age" and "Income."

## Steps to Build a Decision Tree Algorithm

1. **Data Preparation**: Load and preprocess the dataset.
2. **Feature Selection**: Use criteria like Entropy, Information Gain, Gini Index, or Chi-Square to select the most important features.
3. **Tree Construction**: Create nodes and branches based on feature splits.
4. **Prediction**: Use the constructed tree to make predictions on new data.
5. **Evaluation**: Measure the model’s accuracy using metrics like Accuracy Score or Confusion Matrix.

## Feature Selection in Decision Trees

To determine how the data should be split, Decision Trees use various metrics to evaluate the effectiveness of the splits. The commonly used criteria are:

### 1. **Entropy**
Entropy measures the uncertainty or disorder within a dataset. A lower entropy value indicates less uncertainty.

**Formula:**

Entropy(S) = - Σ (p_i * log2(p_i))

Where:
- p_i is the probability of class i in set S.

### 2. **Information Gain**
Information Gain is the reduction in entropy after a split. It helps to determine how well a feature separates the data.

**Formula:**

Information Gain = Entropy(S) - Σ (|S_i| / |S|) * Entropy(S_i)

Where:
- S_i is the subset resulting from the split,
- |S| is the total number of samples in the dataset, and
- |S_i| is the number of samples in the subset S_i.

### 3. **Gini Index**
The Gini Index measures the impurity of a node. A lower Gini Index means the node is more pure.

**Formula:**

Gini Index(S) = 1 - Σ (p_i^2)

Where:
- p_i is the probability of class i in set S.

### 4. **Chi-Square**
The Chi-Square test evaluates the difference between expected and observed frequencies, typically used in categorical data.

**Formula:**

Chi-Square = Σ ((O_i - E_i)^2 / E_i)

Where:
- O_i is the observed frequency of class i,
- E_i is the expected frequency of class i.



## Implementation in Python

### Import Libraries

```python
import numpy as np
import pandas as pd
```

### Load Dataset

```python
data = pd.read_csv('/content/drive/MyDrive/DataSets/kyphosis.csv')
data.head()  # View the first 5 rows
```

### Analyze Dataset

```python
# Shape of the dataset
data.shape

# Information about the dataset
data.info()
```

### Example Dataset

- **y**: Target variable (e.g., "Kyphosis").
- **x**: Feature variables:
  - Age
  - Number
  - Start

### Preprocessing

#### Drop Columns

```python
# X-axis (Features)
x = data.drop('Kyphosis', axis=1)
x.head()  # Check the features after dropping the target column

# Y-axis (Target)
y = data['Kyphosis']
y.head()  # Check the target variable
```

### Split Data

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Check shapes of training and testing datasets
x_train.shape  # Training data shape
x_test.shape   # Testing data shape
```

### Train a Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
model = DecisionTreeClassifier()

# Train the model on training data
model.fit(x_train, y_train)
```

### Make Predictions

```python
# Predict values for test data
predictions = model.predict(x_test)
```

### Evaluate the Model

#### Accuracy Score

```python
from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
```

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)
```

## Detailed Explanation of Code

### Data Import and Preparation:

- `pd.read_csv()`: Loads the dataset from a CSV file.
- `data.head()`: Displays the first 5 rows of the dataset to get an overview of the data.

### Feature Selection:

- `x = data.drop('Kyphosis', axis=1)`: Removes the target column ('Kyphosis') from the dataset to create the feature set.
- `y = data['Kyphosis']`: Defines the target variable which is the column 'Kyphosis'.

### Data Splitting:

- `train_test_split()`: Splits the data into training and testing sets. The parameter `test_size=0.3` means 30% of the data will be used for testing, and the remaining 70% will be used for training.

### Model Initialization and Training:

- `DecisionTreeClassifier()`: Initializes the decision tree classifier.
- `model.fit(x_train, y_train)`: Trains the model on the training dataset.

### Prediction:

- `model.predict(x_test)`: Makes predictions on the test data using the trained model.

### Model Evaluation:

- `accuracy_score()`: Compares the predicted values with the actual values to calculate the accuracy.
- `confusion_matrix()`: Displays the confusion matrix to evaluate the model's performance by showing true positives, false positives, true negatives, and false negatives.


