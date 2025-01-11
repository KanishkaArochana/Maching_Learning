# Evaluate Regression Models in Machine Learning

Evaluating regression models is crucial for understanding how well a model performs in predicting continuous target variables. Several metrics are commonly used to assess the performance of regression models, including **Mean Absolute Error (MAE)**, **Root Mean Square Error (RMSE)**, and **R² Score (Coefficient of Determination)**. This guide explains each metric with examples.

---

## Why Evaluate Regression Models?

1. **Assess Performance**: To measure how close the model's predictions are to the actual values.
2. **Compare Models**: To decide which model performs best for a specific problem.
3. **Identify Limitations**: To understand areas where the model may need improvement.
4. **Optimize Models**: To refine hyperparameters or features to improve accuracy.

---

This guide explains these evaluation metrics with code examples and descriptions of the special methods used.

## Libraries Required

We will use the following Python libraries:

- `numpy` for mathematical operations.
- `pandas` for handling data.
- `sklearn.metrics` for evaluation metrics.

```python
import numpy as np
import pandas as pd
```

## Dataset Example

Here, we have a simple dataset that correlates study hours (`hours`) with exam scores (`score`).

```python
df = pd.DataFrame({
    'hours' : [8, 11, 9, 5, 7.5, 9.5, 10, 7, 9, 9.5, 8, 10.5, 9, 6.5, 9 ],
    'score' : [56, 70, 51, 24, 30, 66, 48, 36, 42, 61, 39, 87, 73, 48, 46]
})
df.head()
```

## Get Actual and Predicted Values

- `y` represents the actual scores.
- `y_pred` represents the predicted scores. In this example, we assume a linear relationship between study hours and score: `score = 8 * hours - 15`.

```python
y = np.array(df.score.values)
y_pred = 8 * df.hours.values - 15
```

### 1. Mean Absolute Error (MAE)

#### What is MAE?

The Mean Absolute Error (MAE) calculates the average of the absolute differences between actual and predicted values. It tells us, on average, how far off the predictions are.

- **Formula**:  
  MAE = (1 / n) * Σ | yᵢ - ŷᵢ |  
  Where:  
  - yᵢ: Actual value  
  - ŷᵢ: Predicted value  
  - n: Total number of samples  

- **Characteristics**:
  - MAE gives equal weight to all errors.
  - The lower the MAE, the better the model.

- **Another Example**:
  Suppose actual values are [10, 20, 30], and predicted values are [12, 18, 33].  
  MAE = (|10 - 12| + |20 - 18| + |30 - 33|) / 3  
  MAE = (2 + 2 + 3) / 3 = 2.33

**Code Explain**:
#### Special Method: `mean_absolute_error`

**Description**: The `mean_absolute_error` method is part of `sklearn.metrics`. It computes the MAE between two arrays, `y_true` (actual values) and `y_pred` (predicted values).

**Syntax**:
```python
mean_absolute_error(y_true, y_pred)
```

**Parameters**:
- `y_true`: Array of actual values.
- `y_pred`: Array of predicted values.

**Returns**: A float representing the MAE.

**Code Example**:
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error:", mae)
```

**Example Output**:
```
Mean Absolute Error: 4.933333333333334
```

### 2. Root Mean Squared Error (RMSE)

#### What is RMSE?

The Root Mean Squared Error (RMSE) measures the square root of the average squared differences between actual and predicted values. It is more sensitive to large errors than MAE.

- **Formula**:  
  RMSE = √[(1 / n) * Σ (yᵢ - ŷᵢ)²]  

- **Characteristics**:
  - RMSE emphasizes larger errors due to squaring.
  - A lower RMSE indicates a better fit.

- **Example**:
  Using the same values [10, 20, 30] and [12, 18, 33]:  
  RMSE = √[((10 - 12)² + (20 - 18)² + (30 - 33)²) / 3]  
  RMSE = √[(4 + 4 + 9) / 3] = √(5.67) ≈ 2.38

 **Code Explain**:
#### Special Method: `mean_squared_error`

**Description**: The `mean_squared_error` method is part of `sklearn.metrics`. It computes the mean squared error (MSE) between two arrays, `y_true` and `y_pred`. RMSE is derived by taking the square root of MSE.

**Syntax**:
```python
mean_squared_error(y_true, y_pred, squared=True)
```

**Parameters**:
- `y_true`: Array of actual values.
- `y_pred`: Array of predicted values.
- `squared`: If True (default), returns MSE. If False, returns RMSE.

**Returns**: A float representing the MSE or RMSE.

**Code Example**:
```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Root Mean Squared Error:", rmse)
```

**Example Output**:
```
Root Mean Squared Error: 6.865199657198665
```

### 3. R2 Score

#### What is R2 Score?

The R2 Score measures how well the regression predictions approximate the actual data. It ranges from 0 to 1:
- 1: Perfect prediction.
- 0: Predictions are no better than the mean of the actual values.
- Negative: Predictions are worse than the mean.

- **Formula**:  
  R² = 1 - (SS_residual / SS_total)  
  Where:  
  - SS_residual = Σ (yᵢ - ŷᵢ)² (Residual sum of squares)  
  - SS_total = Σ (yᵢ - ȳ)² (Total sum of squares)  
  - ȳ: Mean of actual values  

- **Characteristics**:
  - R² ranges from 0 to 1. A value closer to 1 indicates a better fit.
  - A negative R² implies the model performs worse than a horizontal line (mean prediction).

- **Another Example**:
  Using actual values [10, 20, 30] and predicted values [12, 18, 33]:  
  ȳ = (10 + 20 + 30) / 3 = 20  
  SS_total = (10 - 20)² + (20 - 20)² + (30 - 20)² = 100 + 0 + 100 = 200  
  SS_residual = (10 - 12)² + (20 - 18)² + (30 - 33)² = 4 + 4 + 9 = 17  
  R² = 1 - (SS_residual / SS_total) = 1 - (17 / 200) = 0.915

**Code Explain**:

#### Special Method: `r2_score`

**Description**: The `r2_score` method is part of `sklearn.metrics`. It computes the R2 Score for a regression model.

**Syntax**:
```python
r2_score(y_true, y_pred)
```

**Parameters**:
- `y_true`: Array of actual values.
- `y_pred`: Array of predicted values.

**Returns**: A float representing the R2 Score.

**Code Example**:
```python
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print("R2 Score:", r2)
```

**Example Output**:
```
R2 Score: 0.9644574204188665
```

## Summary Table of Metrics

| Metric  | Definition                                           | Best Value | Characteristics                          |
|---------|-----------------------------------------------------|------------|------------------------------------------|
| **MAE** | Mean of absolute differences between predictions and actuals | 0          | Easy to interpret; sensitive to all errors equally. |
| **RMSE**| Square root of mean squared errors                  | 0          | Penalizes larger errors more; harder to interpret. |
| **R²**  | Proportion of variance explained by the model       | 1          | Indicates goodness of fit; closer to 1 is better. |

## Conclusion

- **MAE** tells you the average magnitude of errors.
- **RMSE** penalizes larger errors more heavily.
- **R2 Score** indicates the overall goodness of fit.

Using these metrics together gives a comprehensive understanding of a regression model's performance.
