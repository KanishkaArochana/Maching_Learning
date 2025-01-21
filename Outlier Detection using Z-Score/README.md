
# Outlier Detection using Z-Score

## What is an Outlier?

An outlier is a data point that significantly differs from the other observations in a dataset. These extreme values can distort statistical analyses, leading to inaccurate conclusions. In the context of machine learning and data analysis, removing outliers helps improve model performance by ensuring the model is not unduly influenced by these extreme values.

### Example:

Imagine you have a dataset of student scores in an exam:

| Student | Score |
|---------|-------|
| A       | 70    |
| B       | 80    |
| C       | 85    |
| D       | 95    |
| E       | 100   |
| F       | 500   |

Here, **500** is an outlier as it is far away from the other data points.

---

## Why Using Z-Score for Outlier Detection?

The **Z-Score** is a statistical measurement that describes a data point's relation to the mean of a dataset in terms of standard deviations. A Z-Score tells you how many standard deviations a particular data point is from the mean. 

### Z-Score Formula:

The formula to calculate the Z-Score is:

z = (X - μ) / σ

yaml
Copy code

Where:
- `z` = Z-Score
- `X` = The value (data point)
- `μ` = Mean of the dataset
- `σ` = Standard deviation of the dataset

### Explanation:

- If a Z-Score is **greater than 2** or **less than -2**, the data point is considered an outlier.
- A Z-Score of **0** means the data point is exactly at the mean of the dataset.

---

## Threshold Value

The **threshold value** is a predefined Z-Score that you use to determine if a data point is an outlier. Common threshold values are **2** or **3**, meaning any point with a Z-Score greater than 2 or less than -2 is an outlier.

---

## Example: Outlier Detection using Z-Score

Let's consider the following dataset of exam scores:

| Student | Score |
|---------|-------|
| A       | 70    |
| B       | 80    |
| C       | 85    |
| D       | 95    |
| E       | 100   |
| F       | 500   |

1. **Calculate the Mean and Standard Deviation:**

μ = (70 + 80 + 85 + 95 + 100 + 500) / 6 = 155

swift
Copy code

The standard deviation (`σ`) is calculated as the square root of the variance, where variance is:

σ = sqrt(((X₁ - μ)² + (X₂ - μ)² + ... + (Xₙ - μ)²) / n)

markdown
Copy code

Using the formula, the standard deviation is approximately **158.9**.

2. **Calculate the Z-Score for each data point:**

z_A = (70 - 155) / 158.9 ≈ -0.534 z_B = (80 - 155) / 158.9 ≈ -0.472 z_C = (85 - 155) / 158.9 ≈ -0.440 z_D = (95 - 155) / 158.9 ≈ -0.377 z_E = (100 - 155) / 158.9 ≈ -0.347 z_F = (500 - 155) / 158.9 ≈ 2.17

yaml
Copy code

3. **Compare Z-Scores to Threshold:**

If we use a threshold of 2, then any data point with a Z-Score greater than 2 is an outlier.

In this case, **F (500)** has a Z-Score of 2.17, which is greater than 2, so it is an outlier.

---

## Summary Table:

| Student | Score | Z-Score  | Outlier? |
|---------|-------|----------|----------|
| A       | 70    | -0.534   | No       |
| B       | 80    | -0.472   | No       |
| C       | 85    | -0.440   | No       |
| D       | 95    | -0.377   | No       |
| E       | 100   | -0.347   | No       |
| F       | 500   | 2.17     | Yes      |

In this case, **F (500)** is an outlier.

---

## Code Explain in Outlier Detection using Z-Score

Outlier detection using the Z-score is a statistical method to identify data points that significantly deviate from the rest of the dataset. This method calculates the Z-score for each data point in a feature (or column) and flags data points as outliers if the Z-score is above or below a certain threshold (commonly between -3 and 3).

### 1. Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

- `numpy`: Provides support for numerical calculations.
- `matplotlib.pyplot`: Used for plotting visualizations like histograms.
- `pandas`: Used for data manipulation and analysis.

### 2. Import Dataset

```python
data = pd.read_csv('DataSets/insurance.csv')
```

- `pd.read_csv()`: Reads a CSV file from the specified path and loads it into a pandas DataFrame.

### 3. View First 5 Rows

```python
data.head()
```

- `data.head()`: Displays the first 5 rows of the dataset to get an initial understanding of the data.

### 4. Dataset Shape

```python
data.shape
```

- `data.shape`: Returns the number of rows and columns in the dataset.

## 5. Data Visualization for Charges Column

```python
plt.hist(data['charges'])
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.title('Distribution of Charges')
plt.show()
```

- `plt.hist()`: Plots a histogram to visualize the distribution of the charges column.
- `plt.xlabel()`: Labels the x-axis.
- `plt.ylabel()`: Labels the y-axis.
- `plt.title()`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

### 6. Dataset Summary Statistics

```python
data.describe()
```

- `data.describe()`: Returns the summary statistics of the dataset, such as mean, standard deviation, min, max, and percentiles.

## 7. Calculate Mean of Charges Column

```python
mean = np.mean(data['charges'])
mean
```

- `np.mean()`: Computes the mean (average) of the charges column.

## 8. Calculate Standard Deviation of Charges Column

```python
std = np.std(data['charges'])
std
```

- `np.std()`: Computes the standard deviation of the charges column, which measures the amount of variation or dispersion.

### 9. Calculate Z-Score for Charges Column

```python
(data['charges'] - mean) / std
```

- **Z-Score Formula**: The Z-score is calculated as \( \frac{X - \mu}{\sigma} \), where \( X \) is the data point, \( \mu \) is the mean, and \( \sigma \) is the standard deviation. This formula helps in determining how many standard deviations away a data point is from the mean.

### 10. Add Z-Score to Dataset

```python
data['Charges z-score'] = (data['charges'] - mean) / std
```

- Adds a new column, `Charges z-score`, to the dataset that contains the Z-score for each entry in the charges column.

### 11. Identify Outliers (Z-Score Threshold)

```python
data[data['Charges z-score'] > 3]  # Values greater than 3
data[data['Charges z-score'] < -3]  # Values less than -3
```

- Outliers are identified when the Z-score exceeds a threshold, usually set between 3 and -3. This step filters data points where the Z-score is greater than 3 or less than -3.

### 12. Find Minimum Z-Score Value

```python
data['Charges z-score'].min()
```

- `min()`: Finds the minimum Z-score value in the `Charges z-score` column.

### 13. Find Maximum Z-Score Value

```python
data['Charges z-score'].max()
```

- `max()`: Finds the maximum Z-score value in the `Charges z-score` column.

### 14. Remove Outliers Based on Z-Score

```python
outlier_indexes = []
outlier_indexes.extend(data.index[data['Charges z-score'] > 3].tolist())
outlier_indexes.extend(data.index[data['Charges z-score'] < -3].tolist())
```

- Creates a list, `outlier_indexes`, to store the indexes of rows where the Z-score exceeds 3 or is below -3.
- `data.index`: Retrieves the row indexes where conditions are met.
- `tolist()`: Converts the indexes into a list.

### 15. Remove Outliers from Dataset

```python
new_data = data.drop(data.index[outlier_indexes])
```

- `data.drop()`: Removes the rows from the dataset where the Z-score indicates outliers (rows with indexes stored in `outlier_indexes`).

### 16. Before and After Dataset Shape

```python
data.shape  # Before removing outliers
new_data.shape  # After removing outliers
```

- `data.shape`: Returns the shape of the dataset before removing outliers.
- `new_data.shape`: Returns the shape of the dataset after removing outliers.

### 17. Remove Z-Score Column

```python
if 'Charges z-score' in new_data.columns:
    new_data = new_data.drop('Charges z-score', axis=1)
else:
    print("Column 'Charges z-score' not found in DataFrame.")
```

- **Drop Z-Score Column**: Removes the `Charges z-score` column from the `new_data` DataFrame if it exists.

### 18. Visualize the Data Before and After Removing Outliers

### Before Removing Outliers

```python
plt.hist(data['charges'])
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.title('Distribution of Charges')
plt.show()
```

### After Removing Outliers

```python
plt.hist(new_data['charges'])
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.title('Distribution of Charges')
plt.show()
```

- **Visualization**: Compares the distribution of charges before and after removing the outliers.

## Special Methods in Code

- **Z-Score Calculation**: The Z-score calculation helps in identifying how far a data point is from the mean of the dataset in terms of standard deviations. A Z-score above 3 or below -3 is generally considered an outlier.
- **Outlier Removal**: By filtering out the rows where the Z-score is above 3 or below -3, we effectively remove the extreme values from the dataset.
- **Dataset Visualization**: The histograms help visualize how the charges column's distribution changes before and after removing outliers.




