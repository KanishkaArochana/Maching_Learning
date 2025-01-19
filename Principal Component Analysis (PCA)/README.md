# Principal Component Analysis (PCA) in Machine Learning

Principal Component Analysis (PCA) is a powerful technique used for dimensionality reduction while preserving the variance in the dataset. It helps to simplify the dataset by reducing the number of features without losing important information.

## What is Dimension Reduction?

**Dimension reduction** refers to the process of reducing the number of features (or dimensions) in a dataset while preserving its important characteristics. By reducing the number of features, the data becomes easier to visualize, analyze, and interpret.

### Why Use Dimension Reduction?

- **Reducing the number of features**: Helps simplify the dataset by removing redundant or irrelevant features.
- **Data visualization**: In many cases, high-dimensional data (with many features) cannot be easily visualized. Dimension reduction allows data to be represented in lower-dimensional spaces (2D, 3D) for better understanding.
  
### Two Main Methods for Dimension Reduction:
1. **Feature Selection**:
   - Involves selecting the most important features from the original dataset. 
   - It reduces the number of features by picking the relevant ones based on some criteria.
  
2. **Feature Extraction**:
   - Involves creating new features (principal components) by combining the existing ones.
   - PCA is a technique for feature extraction.

### Difference Between Feature Selection and Feature Extraction

| **Feature Selection**     | **Feature Extraction**   |
|---------------------------|--------------------------|
| Selects a subset of features | Combines existing features to create new ones |
| No new features are created  | New features (principal components) are created |
| Features are kept in their original form | Transforms original features into new ones |
| Used when features are redundant or irrelevant | Used when there is a need to capture underlying patterns in the data |

## Principal Component Analysis (PCA)

PCA is a statistical method used to transform the data into a new coordinate system, where the first few coordinates (principal components) capture most of the variance in the data. This allows us to reduce the number of features while preserving the most important information.

### Step-by-Step Explanation of PCA

Given a dataset with two features (X1 and X2):

```plaintext
X1 = [4, 8, 6, 6]
X2 = [9, 3, 8, 6]
```

We will apply PCA to reduce the dimensionality. The goal is to find new features, principal components (PC1, PC2), that represent the data in a lower dimension.

#### 1. Covariance Matrix (Means)
The first step in PCA is to center the data by subtracting the mean of each feature.

**Step 1.1: Calculate the mean of each feature**
```java
Mean of X1 = (4 + 8 + 6 + 6) / 4 = 6
Mean of X2 = (9 + 3 + 8 + 6) / 4 = 6.5
```

**Step 1.2: Center the data**
Subtract the mean from each feature value:
```plaintext
Centered X1 = [4-6, 8-6, 6-6, 6-6] = [-2, 2, 0, 0]
Centered X2 = [9-6.5, 3-6.5, 8-6.5, 6-6.5] = [2.5, -3.5, 1.5, -0.5]
```
Now the centered data looks like this:
```plaintext
X1' = [-2, 2, 0, 0]
X2' = [2.5, -3.5, 1.5, -0.5]
```

**Step 1.3: Covariance Matrix**
The covariance matrix expresses how the features (X1 and X2) vary with respect to each other. For 2 features, the covariance matrix is a 2x2 matrix:
```plaintext
Cov(X1, X1), Cov(X1, X2)
Cov(X2, X1), Cov(X2, X2)
```
Calculating the covariance:
```plaintext
Cov(X1, X1) = (Σ(X1' * X1')) / (n-1) = 2.5
Cov(X1, X2) = (Σ(X1' * X2')) / (n-1) = 2.0
Cov(X2, X2) = (Σ(X2' * X2')) / (n-1) = 5.0
```
The covariance matrix is:
```lua
Cov = [[2.5, 2.0],
       [2.0, 5.0]]
```

#### 2. Calculate Eigenvalues
Eigenvalues represent the amount of variance explained by each principal component. To calculate the eigenvalues, we solve the characteristic equation:
```css
| Cov - λI | = 0
```
Where λ is the eigenvalue, I is the identity matrix, and Cov is the covariance matrix.

Solving for the eigenvalues (using a numerical method or eigenvalue solver), we get:
```makefile
Eigenvalues: λ1 = 4.0, λ2 = 3.5
```
We choose the highest eigenvalue (λ1 = 4.0) because it explains more variance in the data.

#### 3. Find Normalized Eigenvectors
Eigenvectors correspond to the direction of the new axes (principal components). To find the eigenvectors, we solve the system of linear equations based on the covariance matrix.

Let the eigenvector for λ1 be v1 and for λ2 be v2. After solving, we get the eigenvectors:
```plaintext
Eigenvector for λ1: [0.8, 0.6]
Eigenvector for λ2: [-0.6, 0.8]
```
Normalize these vectors (to unit length):
```plaintext
Normalized Eigenvector for λ1: [0.8, 0.6]
Normalized Eigenvector for λ2: [-0.6, 0.8]
```

#### 4. Derive New Dataset (Principal Components)
Now, we project the original data onto the eigenvectors to obtain the new dataset.

We calculate the projection by multiplying the centered data by the eigenvectors:

**For PC1 (principal component 1):**
```plaintext
PC1 = X1' * Eigenvector1
PC1 = [-2, 2, 0, 0] * [0.8, 0.6]
PC1 = [-1.6, 1.2, 0, 0]
```

**For PC2 (principal component 2):**
```plaintext
PC2 = X2' * Eigenvector2
PC2 = [2.5, -3.5, 1.5, -0.5] * [-0.6, 0.8]
PC2 = [-1.5, 2.8, -0.6, 0.4]
```
The new dataset in terms of principal components (PC1 and PC2) is:
```plaintext
PC1 = [-1.6, 1.2, 0, 0]
PC2 = [-1.5, 2.8, -0.6, 0.4]
```
These new components (PC1, PC2) now represent the data in a reduced dimension (2D in this case).

#### Final Output:
After performing PCA, the dataset is reduced from 2 features (X1, X2) to 2 principal components (PC1, PC2). The principal components are linear combinations of the original features, capturing the most important variance in the data.

## PCA Implementation in Python

### 1. Import Libraries for PCA
```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```
- `%matplotlib inline`: Used to display plots directly in Jupyter notebooks.
- `numpy`: A library for numerical computing, used to create arrays and perform mathematical operations.
- `matplotlib.pyplot`: A library for plotting graphs and visualizing data.

### 2. Dataset Creation
```python
data = np.array([[40,20],
                [55, 30],
                [70, 60],
                [50, 35],
                [45, 40],
                [62, 75],
                [45, 30],
                [68, 80],
                [80, 70],
                [75, 90]])  # First Column: Marks of Maths, Second Column: Marks of Science
```
A simple 2D dataset is created with two features: marks in maths and science.

### 3. Scatter Plot
```python
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Marks of Maths')
plt.ylabel('Marks of Science')
plt.show()
```
This generates a scatter plot to visualize the relationship between maths and science marks.

### 4. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data
```
- `StandardScaler`: Scales the data such that each feature has a mean of 0 and a standard deviation of 1.
- `fit_transform`: Applies scaling to the data and returns the transformed dataset.

### 5. Applying PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=1)  # Reduce to 1 component
pca.fit(scaled_data)
```
- `PCA`: A dimensionality reduction technique that transforms data into a smaller set of uncorrelated components (principal components).
- `n_components=1`: We reduce the data to one principal component.

### 6. Explained Variance Ratio
```python
pca.explained_variance_ratio_
```
This returns the proportion of variance explained by each principal component. In this case, it shows that 93.5% of the variance is captured by the first principal component.

### 7. Transforming Data Using PCA
```python
pca_scaled_data = pca.transform(scaled_data)
```
`transform`: Applies the dimensionality reduction to the scaled data.

### 8. Shape of Data Before and After PCA
```python
scaled_data.shape  # (10, 2)
pca_scaled_data.shape  # (10, 1)
```
Shows the number of rows and columns in the dataset before and after applying PCA. Initially, the dataset had 2 features (2D), and after PCA, it has only 1 feature (1D).

### 9. Scatter Plot Before and After PCA
```python
pca_scaled_data = pca.inverse_transform(pca_scaled_data)

plt.scatter(scaled_data[:,0],scaled_data[:,1])  # Before PCA (2D)
plt.scatter(pca_scaled_data[:,0],pca_scaled_data[:,1])  # After PCA (1D)
plt.xlabel('Marks of Maths')
plt.ylabel('Marks of Science')
plt.show()
```
- Before PCA (2D): The blue points represent the original 2D data.
- After PCA (1D): The orange points represent the data after reducing the dimensions to 1.

### 10. Visualization of Multi-dimensional Data Using PCA
```python
from sklearn.datasets import load_digits
digits = load_digits()
```
`load_digits`: Loads the digits dataset, a set of 8x8 pixel images of handwritten digits.

### 11. Inspecting the Shape of the Digits Dataset
```python
digits.data.shape  # (1797, 64)
```
This shows that there are 1797 images, each represented by 64 features (the pixels in the 8x8 image).

### 12. Visualizing a Digit Image
```python
import matplotlib.pyplot as plt
plt.matshow(digits.images[1])
plt.show()
```
This displays an image of a handwritten digit using matplotlib.

### 13. Inspecting the Target Values
```python
digits.target[1]  # 1
digits.target[33]  # 5
```
The target array contains the actual label for each digit image. For example, the second image corresponds to the digit '1', and the 34th image corresponds to the digit '5'.

### 14. Applying PCA for Dimensionality Reduction (10 Components)
```python
pca = PCA(n_components=10)
new_digits = pca.fit_transform(digits.data)
```
This reduces the original 64 features (pixels) to 10 principal components, capturing most of the variance in the data.

### 15. Shape of New Data After PCA
```python
new_digits.shape  # (1797, 10)
digits.data.shape  # (1797, 64)
```
The dataset is reduced from 64 dimensions to 10 dimensions.

### 16. Converting to 2D for Visualization
```python
pca = PCA(n_components=2)
new_digits = pca.fit_transform(digits.data)
```
This reduces the dataset to just 2 principal components, making it possible to plot the data in a 2D graph.

### 17. Plotting PCA for Visualization
```python
plt.scatter(new_digits[:,0],new_digits[:,1], c=digits.target)  # Class division
plt.xlabel('First Component (PC1)')
plt.ylabel('Second Component (PC2)')
plt.colorbar()  # Use color bar to show class labels
plt.show()
```
This scatter plot shows the reduced 2D data with class labels represented by different colors.

### 18. PCA for Speeding Up Models
The following steps demonstrate how PCA can be used to speed up machine learning models by reducing the number of features:
- The dataset is first scaled using `StandardScaler`.
- The dataset is split into training and testing sets using `train_test_split`.
- A Logistic Regression model is trained and its accuracy is evaluated.

### 19. Model Training Without PCA
```python
model.fit(X_train, y_train)
```
The model is trained using the full dataset with 64 features.

### 20. Accuracy of the Model Without PCA
```python
accuracy_score(y_test, y_pred)
```
The model achieves an accuracy of approximately 94.4% using all features.

### 21. Applying PCA to the Data for Speedup
```python
pca = PCA(n_components=10)
new_data_pca = pca.fit_transform(new_data)
```
PCA is used again, but this time reducing the data to 10 principal components.

### 22. Train and Test on PCA Data
```python
X_train, X_test, y_train, y_test = train_test_split(new_data_pca, digits.target, test_size=0.2)
```
The data after PCA (with 10 features) is split into training and testing sets.

### 23. Model Training with PCA
```python
model.fit(X_train, y_train)
```
The logistic regression model is trained using the reduced data (10 components).

### 24. Accuracy of the Model After PCA
```python
accuracy_score(y_test, y_pred)
```
The model achieves an accuracy of approximately 88.1% after dimensionality reduction using PCA.

This demonstrates how PCA can reduce the dimensionality of a dataset while retaining most of the variance, which can be useful for visualizing data and speeding up machine learning models.


## Summary

- **PCA** reduces the dimensionality of a dataset while retaining most of its variance.
- **Covariance matrix** helps to understand the relationship between features.
- **Eigenvalues** identify the most significant components.
- **Eigenvectors** represent the direction of the new feature space.
- The new dataset is created by projecting the original data onto the new coordinate system.
