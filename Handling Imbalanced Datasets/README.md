# Handling Imbalanced Datasets in Machine Learning

### What Are Imbalanced Datasets?

Imbalanced datasets occur when the number of data points in one or more classes is significantly higher or lower than the others. This imbalance can negatively affect the performance of machine learning models.

### Types of Classification Problems

#### Binary Classification Problem

Involves two classes.

**Example:** Spam vs. Not Spam.

#### Multiclass Classification Problem

Involves more than two classes.

**Example:** Classifying types of animals (e.g., cat, dog, bird).

### Disadvantages of Using Imbalanced Datasets

- **Bias Toward Majority Class:** Models tend to predict the majority class more frequently, leading to poor performance on minority classes.
- **Inaccurate Accuracy Metric:** Accuracy alone can be misleading. Instead, metrics like Precision, Recall, and F1 Score are more informative.

### Examples of Imbalanced Datasets

- **Spam Filtering:** Emails marked as spam (minority) vs. non-spam (majority).
- **Credit Card Fraud Detection:** Fraudulent transactions (minority) vs. legitimate transactions (majority).

### How to Handle Imbalanced Datasets

There are three common methods to handle imbalanced datasets:

#### 1. Undersampling

- **Definition**: Reduces the size of the majority class by randomly selecting a subset of its data points, balancing it with the minority class.

- **Example: 1**
  - Dataset: 90% class A, 10% class B.
  - Remove some class A data points until the dataset becomes 50% class A and 50% class B.

- **Example: 2**

   - **Dataset:** 1,000 legitimate transactions and 50 fraudulent transactions.
   - **Method:** Select 50 random samples from the legitimate transactions to balance the classes.

**Advantage:** Reduces training time and avoids memory issues.

**Disadvantage:** Risk of losing valuable information from the majority class.

#### 2. Oversampling

- **Definition**: Increases the size of the minority class by duplicating its data points or creating synthetic examples.
- **Example: 1**
  - Dataset: 90% class A, 10% class B.
  - Duplicate or create additional class B data points until the dataset becomes 50% class A and 50% class B.

- **Example: 2**

   - **Dataset:** 1,000 legitimate transactions and 50 fraudulent transactions.
   - **Method:** Duplicate samples from the fraudulent transactions to match the legitimate class.

**Advantage:** Ensures all classes are equally represented.

**Disadvantage:** Risk of overfitting due to repeated samples.

#### 3. SMOTE (Synthetic Minority Oversampling Technique)

- **Definition**: Generates synthetic data points for the minority class based on existing data.
- **How It Works**:
  - Takes a data point from the minority class.
  - Identifies its nearest neighbors.
  - Generates a new data point by interpolating between the data point and one of its neighbors.
- **Example: 1**
  - Dataset: 90% class A, 10% class B.
  - Generate synthetic class B data points until the dataset becomes 50% class A and 50% class B.

- **Example: 2**

   - **Dataset:** 1,000 legitimate transactions and 50 fraudulent transactions.
   - **Method:** SMOTE generates new fraudulent data points by interpolating between existing ones.

**Advantage:** Introduces variability in the minority class, reducing overfitting.

**Disadvantage:** Synthetic samples might not accurately represent the true distribution.

### Differences Between Undersampling, Oversampling, and SMOTE

| Method       | Approach                          | Advantages                     | Disadvantages                        |
|--------------|-----------------------------------|--------------------------------|--------------------------------------|
| Undersampling| Reduces majority class data points| Faster training, simpler dataset| Information loss from majority class |
| Oversampling | Duplicates minority class data points| Preserves majority class information| Risk of overfitting                  |
| SMOTE        | Generates synthetic data points   | Diverse minority class examples| Computational overhead               |
---
## How to Handle Imbalanced Datasets

In machine learning, an imbalanced dataset occurs when the classes in the target variable (y) are not represented equally. This can lead to biased models that perform poorly on the minority class. Here, we’ll cover different techniques to handle imbalanced datasets effectively.



To install the `imbalanced-learn` library, you can use the following command in your Python environment:

```python
!pip install imbalanced-learn
```


## Code Explanation


### 1. Importing Libraries

```python
import numpy as np
import pandas as pd
```

- `numpy`: A library for numerical operations.
- `pandas`: A library for data manipulation and analysis.

### 2. Importing the Dataset

```python
data = pd.read_csv('/content/drive/MyDrive/DataSets/kyphosis.csv')
```

This code loads the dataset from a CSV file into a pandas DataFrame called `data`.

### 3. Viewing Random Rows from the Dataset

```python
data.sample(5)
```

This displays 5 random rows from the dataset to help us understand its structure.

### 4. Divide the Dataset into Features and Target Variable

### Defining Features and Target

```python
x = data.drop('Kyphosis', axis=1)
y = data['Kyphosis']
```

- `x`: contains the feature columns: `Age`, `Number`, and `Star`.
- `y`: Contains the target variable `Kyphosis`.

### Viewing Target Variable (y)

```python
y
```

Displays the target variable to examine the distribution of the class labels.

### Counting the Class Distribution in y

```python
y.value_counts()
```

Shows how many instances there are of each class in the target variable (y), which will reveal if the dataset is imbalanced.

### 5. Visualizing the Imbalance

### Plotting the Class Distribution

```python
y.value_counts().plot(kind='bar')
```

This code plots a bar chart of the class distribution in the target variable (y), making it easier to visualize the imbalance.

### 6. Handling Imbalanced Datasets

## 1. Undersampling

Undersampling involves reducing the size of the majority class to match the minority class. This can be done using the `RandomUnderSampler` from the `imblearn` library.

```python
from imblearn.under_sampling import RandomUnderSampler

# Create object
undersample = RandomUnderSampler()

# Fit and transform
x_under, y_under = undersample.fit_resample(x, y)
```

- `RandomUnderSampler()`: This method randomly reduces the number of samples from the majority class to balance the dataset. An object that performs random undersampling.
- `fit_resample(x, y)`: This function resamples the dataset (x, y) and creates a balanced dataset with fewer majority class instances.

### Viewing the Resampled Target (y_under)

```python
y_under.value_counts()
```

This shows the new distribution of the target variable after undersampling.

### Plotting the Resampled Distribution

```python
y_under.value_counts().plot(kind='bar')
```

Plots a bar chart of the resampled class distribution, which should now be balanced.

## 2. Oversampling

Oversampling involves increasing the size of the minority class to match the majority class. This can be done using the `RandomOverSampler` from `imblearn`.

```python
from imblearn.over_sampling import RandomOverSampler

# Create object
oversample = RandomOverSampler()

# Fit and transform
x_over, y_over = oversample.fit_resample(x, y)
```

- `RandomOverSampler()`: This method randomly increases the number of samples in the minority class. An object that performs random oversampling.
- `fit_resample(x, y)`: This function resamples the dataset (x, y) and increases the number of minority class instances.

### Viewing the Resampled Target (y_over)

```python
y_over.value_counts()
```

This shows the new distribution of the target variable after oversampling.

### Plotting the Resampled Distribution

```python
y_over.value_counts().plot(kind='bar')
```

Plots a bar chart of the resampled class distribution, which should now be balanced.

## 3. SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE is an advanced oversampling technique that generates synthetic examples rather than duplicating minority class instances. This can be done using the `SMOTE` class from `imblearn`.

```python
from imblearn.over_sampling import SMOTE

# Create object
smote = SMOTE()

# Fit and transform
x_smote, y_smote = smote.fit_resample(x, y)
```

- `SMOTE()`: An object that performs SMOTE oversampling by generating synthetic instances of the minority class.
- `fit_resample(x, y)`: This function resamples the dataset (x, y) and generates synthetic data points for the minority class.

### Viewing the Resampled Target (y_smote)

```python
y_smote.value_counts()
```

This shows the new distribution of the target variable after applying SMOTE.

### Plotting the Resampled Distribution

```python
y_smote.value_counts().plot(kind='bar')
```

Plots a bar chart of the class distribution after applying SMOTE, which should be balanced.

## Conclusion

In this document, we explored three methods for handling imbalanced datasets in machine learning:

- **Undersampling**: Reduces the majority class to match the size of the minority class.
- **Oversampling**: Increases the minority class to match the size of the majority class.
- **SMOTE**: Generates synthetic data points for the minority class to achieve balance.

Each of these techniques has its advantages and disadvantages. The best approach depends on the specific dataset and problem at hand.

---
