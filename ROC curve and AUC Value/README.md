#  ROC curve and AUC value  in machine Learning
The ROC curve (Receiver Operating Characteristic curve) and the AUC value (Area Under the Curve) are tools used to evaluate the performance of a classification model.

### 1. ROC Curve Basics
The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at different threshold values.

#### True Positive Rate (TPR) (also called sensitivity or recall)
**Formula:**

        TPR = TP / (TP + FN)

- **TP (True Positives)**: The number of positive cases correctly predicted by the model.
- **FN (False Negatives**): The number of positive cases that the model incorrectly predicted as negative.

This measures the proportion of actual positives correctly identified.

#### False Positive Rate (FPR)
**Formula:**

          FPR = FP / (FP + TN)

- **FP (False Positives)**:: The number of negative cases wrongly predicted as positive.
- **TN (True Negatives)**:: The number of negative cases correctly predicted as negative.

This measures the proportion of negatives incorrectly identified as positives.

### 2. Thresholds
A threshold is a decision boundary for predicting classes. For example, if a model outputs probabilities, a threshold of 0.5 might classify probabilities above 0.5 as positive (1) and below 0.5 as negative (0).

By varying the threshold from 0 to 1, we get different TPR and FPR values.

### 3. Plotting the ROC Curve
- At each threshold, compute TPR and FPR.
- Plot FPR (x-axis) versus TPR (y-axis).
- The curve starts at (0, 0) and ends at (1, 1).
- A perfect model will have a point near (0, 1), indicating high TPR and low FPR.

### 4. Area Under the Curve (AUC)
AUC represents the degree of separability between the classes.

**AUC ranges from 0 to 1:**
- AUC = 1.0: Perfect classifier.
- AUC = 0.5: Random guessing (diagonal line on the ROC plot).
- AUC = < 0.5: Worse than random.

### Example 
#### Threshold Values
Thresholds are defined as \([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]\).

#### Compute TPR and FPR
For each threshold, calculate TPR and FPR based on the number of true positives, false positives, false negatives, and true negatives.

#### Plot ROC Curve
- **X-axis:** FPR.
- **Y-axis:** TPR.
- The curve shows the trade-off between TPR and FPR.

#### AUC Value
The AUC value (shaded area under the curve) is calculated. For this example, AUC is 0.75, indicating a good model.



### Table of Results:
| Threshold | TPR  | FPR      | TP | FP | FN | TN |
|-----------|------|----------|----|----|----|----|
| 0.0       | 1.00 | 1.000000 |  4 |  6 |  0 |  0 |
| 0.2       | 0.75 | 0.833333 |  3 |  5 |  1 |  1 |
| 0.4       | 0.75 | 0.666667 |  3 |  4 |  1 |  2 |
| 0.6       | 0.50 | 0.000000 |  2 |  0 |  2 |  6 |
| 0.8       | 0.50 | 0.000000 |  2 |  0 |  2 |  6 |
| 1.0       | 0.00 | 0.000000 |  0 |  0 |  4 |  6 |

### ROC Curve:
![ROC Curve](blob:https://outlook.office.com/80355039-7ae7-4147-aec6-3d3234b0cc6d)

The ROC curve shows the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR).

- **Blue Line (Diagonal)**: Represents a random guessing model (AUC = 0.5).
- **Orange Curve**: Represents the ROC curve for this model, indicating better-than-random performance.


### Summary
- **ROC Curve:** Visualizes the trade-off between sensitivity and specificity.
- **AUC Value:** Measures the model's ability to distinguish between classes. Higher AUC means better performance.

