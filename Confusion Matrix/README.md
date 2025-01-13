
# Confusion Matrix in Machine Learning

## What is a Confusion Matrix?

A Confusion Matrix is a performance measurement tool for machine learning classification models. It provides a summary of prediction results, showing the counts of correct and incorrect predictions categorized by their true labels and predicted labels. The confusion matrix is structured as follows:

## Key Terminology

1. **True Positive (TP)**

   The model correctly predicted the positive class.

   Example: Predicted "pass" and the actual label is "pass."

2. **False Positive (FP) (Type I Error)**

   The model incorrectly predicted the positive class.

   Example: Predicted "pass" but the actual label is "fail."

3. **True Negative (TN)**

   The model correctly predicted the negative class.

   Example: Predicted "fail" and the actual label is "fail."

4. **False Negative (FN) (Type II Error)**

   The model incorrectly predicted the negative class.

   Example: Predicted "fail" but the actual label is "pass."

## Confusion Matrix Layout

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

## Performance Metrics

1. **Accuracy**

   Measures how often the model makes correct predictions.

   Formula:
  Accuracy = (True Positives (TP) + True Negatives (TN)) / (Total Instances (TP + TN + FP + FN))

2. **Precision**

   Measures the percentage of correctly predicted positive cases.

   Formula: 
Precision = True Positives (TP) / (True Positives (TP) + False Positives (FP))

3. **Recall (Sensitivity)**

   Measures the percentage of actual positives correctly identified.

   Formula: 
Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))

4. **F1-Score**

   Harmonic mean of Precision and Recall.

   Formula: 
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## Example Dataset

```python
actual = [1,1,1,0,0,1,0,0,0,1,0,1,1,0,0]  # pass --> 1, fail --> 0
predicted = [0,1,1,0,0,1,1,0,1,1,0,1,1,0,1]
```

## Accuracy Calculation

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(actual, predicted)
print(f"Accuracy: {accuracy}")
```
- `accuracy_score(y_true, y_pred)`:
  - **Input**: Actual labels (y_true), Predicted labels (y_pred)
  - **Output**: The accuracy score as a floating-point number.

## Generating the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(actual, predicted)
print(f"Confusion Matrix:\n{matrix}")
```
- `confusion_matrix(y_true, y_pred)`:
  - **Input**: Actual labels (y_true), Predicted labels (y_pred)
  - **Output**: A 2x2 matrix showing TP, FP, TN, FN counts.

## Taking the Classification Report

```python
import pandas as pd
from sklearn.metrics import classification_report
report = pd.DataFrame(classification_report(actual, predicted, output_dict=True))
print(report)
```
- `classification_report(y_true, y_pred, output_dict=True)`:
  - **Input**: Actual labels (y_true), Predicted labels (y_pred)
  - **Output**: Detailed metrics such as Precision, Recall, F1-Score for each class in a dictionary format. Use `output_dict=False` to get a formatted string.
## Output Examples

### Confusion Matrix:

```
[[TN, FP],
 [FN, TP]]
```
### Output (Example):

#### Confusion Matrix:

```
[[5, 3],
 [1, 6]]
```

- **True Positive (TP)**: 5
- **False Positive (FP)**: 3
- **True Negative (TN)**: 6
- **False Negative (FN)**: 1

### Metrics:

- **Accuracy**: 0.733
- **Precision**: 0.667
- **Recall**: 0.875
- **F1-Score**: 0.75

### Classification Report:

```
|                | 0          | 1          | accuracy  | macro avg | weighted avg |
|----------------|------------|------------|-----------|-----------|--------------|
| precision      | 0.833333   | 0.666667   | 0.733333  | 0.750000  | 0.755556     |
| recall         | 0.625000   | 0.857143   | 0.733333  | 0.741071  | 0.733333     |
| f1-score       | 0.714286   | 0.750000   | 0.733333  | 0.732143  | 0.730952     |
| support        | 8.000000   | 7.000000   | 0.733333  | 15.000000 | 15.000000    |
```

