# Feature Selection in Unsupervised Learning

Feature selection in unsupervised learning helps reduce redundancy and improve the quality of clustering or pattern detection by removing less useful or highly correlated features. We'll discuss two techniques: Variance Thresholding and Correlation Analysis, with step-by-step explanations using the provided dataset and heatmap.



### Example Dataset:
| Name | Math | Chemistry | Maths | Physics | General Test |
|------|------|-----------|-------|---------|--------------|
| A    | 70   | 60        | 70    | 50      | 70           |
| B    | 60   | 80        | 60    | 50      | 70           |
| C    | 40   | 65        | 40    | 50      | 60           |
| D    | 80   | 55        | 80    | 50      | 60           |
| E    | 30   | 60        | 30    | 50      | 80           |

## 1. Variance-Based Feature Selection
Variance is a measure of how spread out the values in a column are. Features with low variance contain less information and may not help the model.

### Variance Thresholding

Variance Thresholding is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e., features that have the same value in all samples.

**Formula for Variance:**

Variance = Σ(x_i - μ)² / n


Where:
- x_i = data point
- μ = mean of the feature
- n = number of data points

**Step-by-Step:**
1. Calculate the mean for each feature (column).

   - For Math: Mean = \((70 + 60 + 40 + 80 + 30) / 5 = 56\)
   - For Chemistry: Mean = \((60 + 80 + 65 + 55 + 60) / 5 = 64\)
   - For Maths: Mean = \((70 + 60 + 40 + 80 + 30) / 5 = 56\)
   - For Physics: Mean = \((50 + 50 + 50 + 50 + 50) / 5 = 50\)
   - For General Test: Mean = \((70 + 70 + 60 + 60 + 80) / 5 = 68\)

2. Calculate the variance for each feature:

   - Variance for Math: 
    Here's your data converted to Markdown format:


| Feature       | Variance |
|---------------|----------|
| Math          | 344      |
| Chemistry     | 74       |
| Maths         | 344      |
| Physics       | 0        |
| General Test  | 56       |



**Interpretation:**
If the variance is low (close to 0), it indicates that the feature has little variation across data points and does not provide much information. For example, the Physics column has no variation (all values are 50), so it would be removed based on low variance.

## 2. Correlation-Based Feature Selection
Correlation measures the relationship between two features. A correlation close to 1 or -1 indicates a strong relationship, meaning the features are highly related and provide redundant information. In such cases, one feature can be removed to reduce redundancy.

**Formula for Pearson Correlation Coefficient:**
Correlation(X, Y) = Cov(X, Y) / (σ_X * σ_Y)


Where:
- Cov(X, Y) is the covariance between features X and Y,
- σ 
X
​
 ,σ 
Y
​
   are the standard deviations of X and Y.

**Step-by-Step:**
1. Calculate the covariance between each pair of features (like Math and Chemistry, Math and General Test, etc.).

2. Calculate the correlation for each pair of features. From the given heatmap:

   - Math and Chemistry: Correlation = -0.15 (weak negative correlation).
   - Math and Maths: Correlation = 1.0 (perfect positive correlation). This suggests that the Maths feature is highly redundant with the Math feature, and one of them should be removed.
   - Math and General Test: Correlation = -0.49 (moderate negative correlation).
   - Chemistry and General Test: Correlation = 0.12 (weak positive correlation).


| Feature 1 | Feature 2      | Correlation Value |
|-----------|----------------|-------------------|
| Math      | Chemistry      | -0.15             |
| Math      | Maths          | 1.0               |
| Math      | General Test   | -0.49             |
| Chemistry | General Test   | 0.12              |


**Interpretation:**
Highly correlated features (correlation close to 1 or -1) are usually redundant. For example, Math and Maths have a perfect correlation of 1.0, so you would typically remove one of these features because they contain nearly the same information.

### Why Remove Low-Variance or Highly Correlated Features?
- **Low Variance:** Features with low variance (little to no variation) do not contribute meaningful information to the model. Removing them helps reduce the dimensionality and makes the model more efficient.
- **Highly Correlated Features:** If two features are highly correlated, they provide almost the same information. Retaining both can lead to overfitting and unnecessary complexity in the model. Removing one of them simplifies the model without losing important information.

## Code Explanation

## 1. Variance 

### Step 1: Importing the Dataset

```python
import pandas as pd

data = pd.DataFrame({'Math':[70, 60, 40, 80, 30],
                     'Chemistry': [60, 80, 65, 55, 60],
                     'Maths':[70, 60, 40, 80, 30],
                     'Physics': [50, 50, 50, 50, 50],
                     'General_Test': [70, 70, 60, 60, 80]})

print(data)
```

- This code creates a DataFrame `data` using pandas, which contains information about students' scores in five subjects.
- `pd.DataFrame()` is used to create a DataFrame from a dictionary, where the keys represent the column names and the values are lists of scores.
- `print(data)` displays the DataFrame.

### Step 2: Applying Variance Threshold for Feature Selection

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0)  # threshold=0 means no features are dropped unless they have zero variance

selected_features = selector.fit_transform(data)
```

- `VarianceThreshold` is used to remove features with low variance.
- The `threshold=0` means that any feature with zero variance will be removed. Variance is a measure of how much the values in a column differ from the mean; if all values are the same, variance will be 0.
- `fit_transform()` applies the selector and returns the transformed data, which only contains features with variance above the threshold.

### Step 3: Displaying the Transformed Data

```python
selected_features
```

- This displays the transformed dataset after applying the variance threshold. The Physics column, which had zero variance (all values were 50), is removed.

### Step 4: Convert Back to DataFrame

```python
data = pd.DataFrame(selected_features, columns=selector.get_feature_names_out())
data
```

- Converts the transformed array back to a pandas DataFrame.
- `get_feature_names_out()` retrieves the names of the remaining features after the transformation.
## 2. Correlation
### Step 5: Pearson Correlation Calculation

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cor = data.corr()
cor
```

- This imports necessary libraries for correlation calculation and visualization: numpy, seaborn, and matplotlib.
- `data.corr()` computes the Pearson correlation coefficient between each pair of features in the data. The result is a correlation matrix that shows the relationships between features.

### Step 6: Visualization of Correlation Matrix

```python
plt.figure(figsize=(8, 6))
sns.heatmap(cor, annot=True, cmap='Wistia')
plt.show()
```

- The code visualizes the correlation matrix as a heatmap using `seaborn.heatmap()`.
- `annot=True` adds the correlation values in the heatmap.
- `cmap='Wistia'` specifies the color palette.
- `plt.show()` displays the heatmap.

### Step 7: Identifying Highly Correlated Features

```python
corr_features = set()
for i in range(len(cor.columns)):
  for j in range(i):
    if abs(cor.iloc[i, j]) > 0.9:  # threshold for high correlation
      corr_features.add(cor.columns[i])
```

- This code identifies features that have a high correlation (greater than 0.9 in absolute value).
- It iterates over the correlation matrix and adds the column names of highly correlated features to the `corr_features` set.

### Step 8: Displaying Highly Correlated Features

```python
corr_features
```

- This displays the set of features with high correlation. In this case, it outputs `{'Maths'}` as "Math" and "Maths" are highly correlated.

### Step 9: Removing Highly Correlated Features

```python
data = data.drop(corr_features, axis=1)
data
```

- The `drop()` method removes the highly correlated features identified in the previous step (in this case, the "Maths" column).
- `axis=1` specifies that columns should be dropped.

### Final Step: The Dataset after Removal

- The final dataset (`data`) is now free of features with low variance and high correlation, which reduces redundancy and improves the model's performance by focusing on more informative features.


### Summary:
- **Variance-based selection** removes features with little variation, as they do not add meaningful information.
- **Correlation-based selection** removes highly correlated features, as they provide redundant information, leading to a more efficient model.

