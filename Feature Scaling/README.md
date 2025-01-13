
# Feature Scaling in Machine Learning

Feature scaling is a technique used to standardize the range of independent variables or features of data. In machine learning, scaling features ensures that the algorithm can process the data more efficiently, especially when dealing with models that rely on the magnitude of data (like linear regression, neural networks, etc.).

## Why Use Feature Scaling?

- **Increase Accuracy**: Many machine learning algorithms perform better when the data is scaled. Models like gradient descent converge faster because they are more sensitive to the scale of data.
- **Increase Speed**: Feature scaling can make the training process faster. When features are on the same scale, the algorithms can converge more quickly, reducing the time to train the model.

## Methods of Feature Scaling

There are two primary methods for feature scaling:

### 1. Normalization (Min-Max Scaling)

Normalization, or Min-Max scaling, is a technique that transforms the data into a specific range, typically between 0 and 1. This method works by subtracting the minimum value of the feature and then dividing by the range (max - min).

**Formula:**

```
Normalized Value = (X - min(X)) / (max(X) - min(X))
```

**Manually Calculated Example:**

Let's manually calculate the normalized values for the feature "age" from the dataset:

**Dataset:**

| Age | Salary |
|-----|--------|
| 26  | 50000  |
| 29  | 70000  |
| 34  | 55000  |
| 31  | 41000  |

Minimum age value = 26
Maximum age value = 34

For the normalization of age:

```
Normalized Age for 26 = (26 - 26) / (34 - 26) = 0 / 8 = 0
Normalized Age for 29 = (29 - 26) / (34 - 26) = 3 / 8 = 0.375
Normalized Age for 34 = (34 - 26) / (34 - 26) = 8 / 8 = 1
Normalized Age for 31 = (31 - 26) / (34 - 26) = 5 / 8 = 0.625
```

So, the normalized age values will be:
[0, 0.375, 1, 0.625]

**Example Code (Normalization):**

```python
# Import numpy library
import numpy as np

# Example dataset
data = np.array([[26, 50000], 
                 [29, 70000], 
                 [34, 55000], 
                 [31, 41000]])  # Column names: age, salary

# Import MinMaxScaler from sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

# Create MinMaxScaler object
scaler = MinMaxScaler()

# Apply fit_transform method to scale the data
scaled_data = scaler.fit_transform(data)

# Output the scaled data (Values between 0 and 1)
print(scaled_data)
```

### 2. Standardization (Z-Score Normalization)

Standardization is a method that scales the data so that it has a mean of 0 and a standard deviation of 1. This method is useful when the features have varying scales, and the algorithm assumes the data is normally distributed.

**Formula:**

```
Standardized Value = (X - μ) / σ
```

Where:

- X is the original value
- μ is the mean of the feature
- σ is the standard deviation of the feature

**Manually Calculated Example:**

Let's manually calculate the standardized values for the feature "age" from the dataset:

Mean age (μ) = (26 + 29 + 34 + 31) / 4 = 30

Standard deviation (σ):

```
σ = √((Σ(X - μ)²) / N) = √(((26 - 30)² + (29 - 30)² + (34 - 30)² + (31 - 30)²) / 4)
  = √((16 + 1 + 16 + 1) / 4) = √(34 / 4) = 2.91
```

For the standardization of age:

```
Standardized Age for 26 = (26 - 30) / 2.91 = -4 / 2.91 = -1.38
Standardized Age for 29 = (29 - 30) / 2.91 = -1 / 2.91 = -0.34
Standardized Age for 34 = (34 - 30) / 2.91 = 4 / 2.91 = 1.38
Standardized Age for 31 = (31 - 30) / 2.91 = 1 / 2.91 = 0.34
```

So, the standardized age values will be:
[-1.38, -0.34, 1.38, 0.34]

**Example Code (Standardization):**

```python
# Import StandardScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

# Create StandardScaler object
scaler = StandardScaler()

# Apply fit_transform method to scale the data
scaled_data = scaler.fit_transform(data)

# Output the scaled data (Mean 0, Standard Deviation 1)
print(scaled_data)
```

## Library Methods & Special Functions

### MinMaxScaler (Normalization)

The MinMaxScaler is a class from sklearn.preprocessing used for Normalization (Min-Max scaling). It scales the data such that each feature is transformed into a fixed range, typically [0, 1]. This method is sensitive to outliers since it relies on the minimum and maximum values of the feature.

**Usage:** MinMaxScaler() is used to create an object that scales the data within the range [0, 1] using the Min-Max formula described above.

### StandardScaler (Standardization)

The StandardScaler is a class from sklearn.preprocessing used for Standardization (Z-Score normalization). It transforms the data such that each feature has a mean of 0 and a standard deviation of 1.

**Usage:** StandardScaler() is used to create an object that standardizes the data, i.e., it scales the data based on the mean and standard deviation of the feature.

### scaler.fit_transform()

The fit_transform() method is used in both MinMaxScaler and StandardScaler to perform both the fitting (calculating the required statistics, like the minimum, maximum, mean, and standard deviation) and the transformation (scaling the data) in one step.

- **fit():** This method calculates the necessary statistics (mean, standard deviation, min, max) from the data.
- **transform():** This method applies the scaling based on the statistics obtained from the fit() method.
- **fit_transform():** This is a convenience method that combines both fit() and transform(), making it easier to scale data in one step.

**Example of fit_transform():**

```python
# Create MinMaxScaler object
scaler = MinMaxScaler()

# Apply fit_transform method to scale the data
scaled_data = scaler.fit_transform(data)

# Output the scaled data
print(scaled_data)
```

## Advantages of Feature Scaling

- **Improves Accuracy:** Many algorithms, such as support vector machines (SVM), k-nearest neighbors (KNN), and neural networks, require features to be scaled in order to perform optimally. Without scaling, models can give inaccurate results.
- **Faster Convergence:** Scaling helps in reducing the number of iterations required for an algorithm to converge, thus speeding up training and making the learning process more efficient.
- **Reduces Bias:** Feature scaling helps ensure that no feature dominates or distorts the model due to differing ranges of values.

## Conclusion

Feature scaling is a critical step in the preprocessing phase of machine learning. Whether using Normalization or Standardization, it is essential to apply appropriate scaling based on the algorithm you are using. Normalization is ideal for models that require bounded data, while standardization is best for models that assume a Gaussian distribution.
