
# Train Test Split in Machine Learning

## Why Divide the Full Dataset into Train and Test Sets?

In machine learning, we divide the dataset into training and testing subsets to:

- **Train the Model**: Use the training set to let the model learn patterns and relationships.
- **Evaluate the Model**: Use the test set to measure the model's performance on unseen data, ensuring it generalizes well.

Data is split randomly to avoid bias and ensure a representative sample for both sets.

## How to Ensure the Model's Accuracy

To take care of the model's accuracy:

- Use a representative dataset.
- Avoid overfitting by validating with unseen data (e.g., the test set).
- Use metrics like accuracy, precision, recall, and F1-score to evaluate performance.
- Consider techniques like cross-validation for robust evaluation.

## Implementation of Train-Test Split

### Import Required Library

```python
import numpy as np # For creating and handling numerical arrays
from sklearn.model_selection import train_test_split # For splitting the dataset
```

### Dataset

```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
```

### Separate Data into Train and Test Sets

The `train_test_split` function divides the dataset into four categories:

- Training set for x values (`x_train`).
- Testing set for x values (`x_test`).
- Training set for y values (`y_train`).
- Testing set for y values (`y_test`).

#### Syntax

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
```

#### Parameters:

- **test_size**: A float between 0 and 1 representing the proportion of the dataset to include in the test split.
  - Example: If `test_size=0.2`, 80% of data will go to `x_train` and 20% to `x_test`.
- **random_state**: Ensures reproducibility by controlling the random shuffling of data.
- **shuffle**: Set to False to disable shuffling of data before splitting (default is True).

### Example Code

```python
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Check X values for training
print("x_train:", x_train) # Feature values for training
print("Length of x_train:", len(x_train)) # Number of training samples

# Check X values for testing
print("x_test:", x_test) # Feature values for testing
print("Length of x_test:", len(x_test)) # Number of testing samples

# Check Y values for training
print("y_train:", y_train) # Labels for training
print("Length of y_train:", len(y_train)) # Number of training labels

# Check Y values for testing
print("y_test:", y_test) # Labels for testing
print("Length of y_test:", len(y_test)) # Number of testing labels
```

### Special Methods in `train_test_split`

- **test_size**:
  - Controls the size of the test set. Example: `test_size=0.25` splits 25% of data for testing.
- **random_state**:
  - Ensures consistent results every time you run the code by fixing the randomization.
- **shuffle**:
  - Decides whether to shuffle data before splitting. Default is True.

### Output Example

For the given dataset and `test_size=0.25`, the split might look like this:

```python
x_train: [8 5 11 16 18 20 15 1 4 12 19 13 7 17 6]
x_test: [3 9 10 14 2]
Length of x_train: 15
Length of x_test: 5

y_train: [0 1 1 0 1 1 1 0 0 1 1 1 0 1 0]
y_test: [1 0 0 0 0]
Length of y_train: 15
Length of y_test: 5
```


## Key Points Explained in Comments

### Library Usage:
- `numpy` is used for creating numerical arrays (`x` and `y` in this case).
- `train_test_split` from `sklearn` is used to split the data into training and testing subsets.

### Dataset:
- `x` represents the input features (independent variables).
- `y` represents the target labels (dependent variable or output).

### Splitting:
- The `train_test_split` function divides `x` and `y` into training (`x_train`, `y_train`) and testing (`x_test`, `y_test`) subsets.

### Parameters:
- `test_size=0.25`: 25% of data goes into the test set, and 75% into the training set.
- `random_state=42`: Ensures the split is consistent each time the code runs.

### Output:
- Displays the training and testing datasets along with their lengths, showing how the data was split.
