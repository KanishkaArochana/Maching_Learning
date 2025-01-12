
# One Hot Encoding in Machine Learning

One Hot Encoding is a method used to convert categorical data into numerical format. It is particularly useful for algorithms that cannot handle categorical features directly. For example, machine learning algorithms often require inputs to be numeric, so converting categories into a format that preserves the information in a machine-readable way is crucial.

## Why Use One Hot Encoding?

- **Machine Readability**: Many machine learning models cannot interpret categorical data directly. One Hot Encoding transforms such data into a numeric format.
- **Avoid Ordinal Relationships**: Unlike Label Encoding, One Hot Encoding prevents unintended ordinal relationships by creating binary (0/1) columns for each category.
- **Flexibility**: It works well for both small and large datasets, where categorical columns have distinct classes.

## Example Dataset

```python
import pandas as pd

# Sample Dataset
df = pd.DataFrame({
    'result': ['pass', 'fail', 'pass', 'pass', 'absent', 'fail', 'fail', 'pass', 'pass', 'absent', 'pass']
})

df
```
**Explanation of Code:**

- `LabelEncoder` from `sklearn.preprocessing` is used to convert the result column values into numerical values.
- The `fit_transform` method is used to learn the encoding and then transform the data.
- The output will be numerical values representing each category (e.g., absent = 0, fail = 1, pass = 2).


## Dataset Output

| Index | result |
|-------|--------|
| 0     | pass   |
| 1     | fail   |
| 2     | pass   |
| 3     | pass   |
| 4     | absent |
| 5     | fail   |
| 6     | fail   |
| 7     | pass   |
| 8     | pass   |
| 9     | absent |
| 10    | pass   |

## Label Encoding

### Step 1: Assign the result column to a variable

```python
result_category = df['result']
```

### Step 2: Use LabelEncoder from sklearn

Label Encoding converts the categories into numeric values. However, this encoding may inadvertently introduce ordinal relationships.

```python
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder object
obj = LabelEncoder()

# Fit and transform the data
result = obj.fit_transform(result_category)

# Display encoded values
result
```
**Explanation of Code:**

- `LabelBinarizer` from `sklearn.preprocessing` is used for one hot encoding.
- `fit_transform` method is applied to the `result_category` to convert it into binary columns.
- The `obj.classes_` displays the categories as they are ordered in the columns (e.g., ['absent', 'fail', 'pass']).


### Output:

| Index | result (Encoded) |
|-------|------------------|
| 0     | 2                |
| 1     | 1                |
| 2     | 2                |
| 3     | 2                |
| 4     | 0                |
| 5     | 1                |
| 6     | 1                |
| 7     | 2                |
| 8     | 2                |
| 9     | 0                |
| 10    | 2                |

### Note:

The categories are assigned as follows:

- absent = 0
- fail = 1
- pass = 2

While this method is simple, it may introduce a sense of ranking between categories (e.g., absent < fail < pass).

## One Hot Encoding

### Why Use One Hot Encoding?

One Hot Encoding solves the problem of ordinal relationships by converting categories into binary columns. Each column represents whether a particular category is present (1) or not (0).

### Implementation:

```python
from sklearn.preprocessing import LabelBinarizer

# Create LabelBinarizer object
obj = LabelBinarizer()

# Fit and transform the data
result = obj.fit_transform(result_category)

# Display one-hot encoded values
result
```

### Output:

| Index | absent | fail | pass |
|-------|--------|------|------|
| 0     | 0      | 0    | 1    |
| 1     | 0      | 1    | 0    |
| 2     | 0      | 0    | 1    |
| 3     | 0      | 0    | 1    |
| 4     | 1      | 0    | 0    |
| 5     | 0      | 1    | 0    |
| 6     | 0      | 1    | 0    |
| 7     | 0      | 0    | 1    |
| 8     | 0      | 0    | 1    |
| 9     | 1      | 0    | 0    |
| 10    | 0      | 0    | 1    |

### Display Classes:

```python
# Display class labels
obj.classes_
```

### Output:

['absent', 'fail', 'pass']

### Column Mapping:

- 1st column (absent) indicates if the result is "absent."
- 2nd column (fail) indicates if the result is "fail."
- 3rd column (pass) indicates if the result is "pass."

## Summary

- Label Encoding is straightforward but may introduce ordinal relationships between categories.
- One Hot Encoding ensures each category is represented independently, avoiding ranking issues.
- Use the LabelBinarizer class from sklearn for One Hot Encoding.

Both methods are essential tools for preparing categorical data for machine learning tasks.
"""
## Key Differences Between Label Encoding and One Hot Encoding

| Aspect               | Label Encoding                              | One Hot Encoding                              |
|----------------------|---------------------------------------------|-----------------------------------------------|
| Data Representation  | Converts categories into integers           | Converts categories into binary columns       |
| Usage                | Useful for ordinal data where category order matters | Useful for nominal data where categories do not have an order |
| Model Compatibility  | Works well with models that handle ordinal relationships (e.g., decision trees) | Works well for models that cannot interpret categorical data directly (e.g., linear regression) |