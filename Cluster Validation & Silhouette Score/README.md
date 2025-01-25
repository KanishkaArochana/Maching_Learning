# Cluster Validation & Silhouette Score (Unsupervised Learning)

## What is Clustering?
Clustering is a type of unsupervised learning, a method used in machine learning where the data has no predefined labels. The goal is to group similar data points into clusters, where:

- Data points in the same cluster are more similar to each other.
- Data points in different clusters are more distinct.

### Examples of Clustering Algorithms

- **K-Means**: Divides the data into \( k \) clusters by minimizing the distance between data points and their cluster centers.
- **Hierarchical Clustering**: Builds a tree (dendrogram) by merging or splitting clusters step by step.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Groups data points based on their density and can identify outliers.

### Why Not Use Metrics like F1-Score, Precision, Recall in Unsupervised Learning?

These metrics (e.g., F1-Score, Recall, Confusion Matrix) are for supervised learning, where the data has labels. In unsupervised learning:

- **No Labels**: Since there’s no predefined "correct cluster," these metrics can’t be calculated.
- Instead, clustering validation metrics (like Silhouette Score) are used to measure clustering performance.

### Good Clustering Properties

- **Intra-cluster Distance (Cohesion)**:
  - The distance between data points within the same cluster. The smaller this distance, the better the clustering, as it indicates that the data points are close to each other, which means they belong to the same group.
  - Smaller values = Better clustering (points are tightly packed).

- **Inter-cluster Distance (Separation)**:
  - The distance between different clusters. The larger this distance, the better the clustering, as it suggests that the clusters are distinct and easy to separate from one another.
  - Larger values = Better clustering (clusters are more distinct).

### What is Cluster Validation?

Cluster validation evaluates the performance of clustering in unsupervised learning (where no labels exist). One common technique for this is the Silhouette Score.

### Silhouette Score

The Silhouette Score measures how well data points are clustered by comparing:

- **Cohesion (a)**: The average distance of a point to other points in the same cluster.
- **Separation (b)**: The average distance of a point to points in the nearest neighboring cluster.

#### Formula for Silhouette Coefficient (S):

For each data point:

              S = (b - a) / max(a, b)

- \( a \): Cohesion (distance within the same cluster).
- \( b \): Separation (distance to the nearest other cluster).
- \( S \): Silhouette coefficient for a single data point.

The Silhouette Score for the whole dataset is the average of \( S \) across all points.

#### Silhouette Score Range

               Score∈[−1,+1]

- **Near +1**: Points are well clustered (good clustering).
- **Near -1**: Points are poorly clustered (wrong clustering).
- **Around 0**: Points are on the boundary between clusters.



### How Does the Silhouette Score Work?

Given a set of points, the Silhouette Score is calculated based on these steps:

1. **Calculate the distance between points**: You can use various distance metrics, such as:
   - **Euclidean distance**: Measures the straight-line distance between two points.
   - **Manhattan distance**: Measures the sum of the absolute differences of their coordinates.

2. **Calculate the cohesion (a) for each point**: This is the average distance between a point and all other points in the same cluster.

3. **Calculate the separation (b) for each point**: This is the distance between a point and the points in the closest neighboring cluster.

4. **Silhouette Coefficient for each point**: For a given point \(i\), the Silhouette Coefficient \(S(i)\) is calculated as:
   
               S = (b(i) - a(i)) / max(a(i), b(i))

   - \(a(i)\) is the cohesion for point \(i\).
   - \(b(i)\) is the separation for point \(i\).

5. **Silhouette Score**: The average of the Silhouette Coefficients across all points gives the overall Silhouette Score for the clustering. This value ranges from -1 to +1.

## Silhouette Score Calculation Example

We will calculate the Silhouette Score for a small set of data points and clusters.

### Given Data Points:
- **x =** [0.2, 0.6, 0.3, 0.7]
- **y =** [0.9, 0.3, 0.8, 0.2]

### Clusters:
- **Cluster 1**: Points (0.2, 0.9) and (0.3, 0.8)
- **Cluster 2**: Points (0.6, 0.3) and (0.7, 0.2)

### Step 1: Calculate the Distances Between Points

We will use the **Euclidean distance** formula to calculate the distance between points.

Euclidean Distance Formula:
d = √((x2 - x1)² + (y2 - y1)²)

For example, to calculate the distance between points (0.2, 0.9) and (0.3, 0.8):
d = √((0.3 - 0.2)² + (0.8 - 0.9)²) = √(0.01 + 0.01) = √0.02 ≈ 0.141

Now, calculate all the distances between points:

| Points               | Distance |
|----------------------|----------|
| (0.2, 0.9) & (0.3, 0.8)  | 0.141    |
| (0.2, 0.9) & (0.6, 0.3)  | 0.721    |
| (0.2, 0.9) & (0.7, 0.2)  | 0.925    |
| (0.3, 0.8) & (0.6, 0.3)  | 0.707    |
| (0.3, 0.8) & (0.7, 0.2)  | 0.894    |
| (0.6, 0.3) & (0.7, 0.2)  | 0.141    |

### Step 2: Calculate the Cohesion (a) for Each Data Point

The **cohesion** for a point is the average distance to all other points in the same cluster.

For Point (0.2, 0.9) in Cluster 1:
- Distances to other points in Cluster 1: [0.141]
- **Cohesion a = 0.141**

For Point (0.3, 0.8) in Cluster 1:
- Distances to other points in Cluster 1: [0.141]
- **Cohesion a = 0.141**

For Point (0.6, 0.3) in Cluster 2:
- Distances to other points in Cluster 2: [0.141]
- **Cohesion a = 0.141**

For Point (0.7, 0.2) in Cluster 2:
- Distances to other points in Cluster 2: [0.141]
- **Cohesion a = 0.141**

### Step 3: Calculate the Separation (b) for Each Data Point

The **separation** for a point is the average distance to all points in the nearest neighboring cluster.

For Point (0.2, 0.9) in Cluster 1:
- Distances to points in Cluster 2: [0.721, 0.925]
- **Separation b = (0.721 + 0.925) / 2 = 0.823**

For Point (0.3, 0.8) in Cluster 1:
- Distances to points in Cluster 2: [0.707, 0.894]
- **Separation b = (0.707 + 0.894) / 2 = 0.8005**

For Point (0.6, 0.3) in Cluster 2:
- Distances to points in Cluster 1: [0.721, 0.707]
- **Separation b = (0.721 + 0.707) / 2 = 0.714**

For Point (0.7, 0.2) in Cluster 2:
- Distances to points in Cluster 1: [0.925, 0.894]
- **Separation b = (0.925 + 0.894) / 2 = 0.9095**

### Step 4: Calculate the Silhouette Coefficient (S) for Each Point

The **Silhouette Coefficient** for each point is calculated as:

S = (b - a) / max(a, b)

For Point (0.2, 0.9):
- S = (0.823 - 0.141) / max(0.141, 0.823) = 0.682 / 0.823 ≈ 0.830

For Point (0.3, 0.8):
- S = (0.8005 - 0.141) / max(0.141, 0.8005) = 0.6595 / 0.8005 ≈ 0.825

For Point (0.6, 0.3):
- S = (0.714 - 0.141) / max(0.141, 0.714) = 0.573 / 0.714 ≈ 0.803

For Point (0.7, 0.2):
- S = (0.9095 - 0.141) / max(0.141, 0.9095) = 0.7685 / 0.9095 ≈ 0.846

### Step 5: Calculate the Average Silhouette Score

The **average Silhouette Score** is the average of the individual scores:

Average Silhouette Score = (0.830 + 0.825 + 0.803 + 0.846) / 4 ≈ 0.826

### Example Conclusion

The average Silhouette Score for this clustering is **0.826**, indicating a good clustering result with points well-separated between clusters and cohesive within each cluster.

# Code Explanation

### 1. Import Libraries

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')  # use ignore Warning
```

- `pandas `: Used for data manipulation and analysis (e.g., creating and working with data frames).
-  `numpy `: A library for numerical operations, although it’s not actively used in this code.
-  `matplotlib.pyplot `: A library for creating visualizations (e.g., scatter plots).
-  `warnings.filterwarnings('ignore') ` suppresses any warnings that might appear during execution (e.g., deprecated warnings or others).

### 2. Create Dataset

```python
data = pd.DataFrame({
    'x' : [0.2, 0.6, 0.3, 0.7],
    'y' : [0.9, 0.3, 0.8, 0.2]
})

data
```

- This creates a small dataset (`data`) with two features:  `'x'` and  `'y' `.
- `pd.DataFrame()` creates a DataFrame, which is a 2-dimensional structure with rows and columns. The data here contains 4 points (pairs of x, y coordinates).

### 3. Visualization of Data

```python
plt.scatter(data['x'], data['y'])
plt.show()
```

- This visualizes the dataset using a scatter plot, plotting x values on the horizontal axis and y values on the vertical axis.
- `plt.scatter()` creates the scatter plot, and `plt.show()` displays it.

### 4. K-Means Clustering

```python
import sklearn.cluster as cluster

model = cluster.KMeans(n_clusters=2)

model.fit(data)

label = model.predict(data)

label
```

- `import sklearn.cluster as cluster`: Imports the clustering module from the sklearn (Scikit-learn) library.
- `model = cluster.KMeans(n_clusters=2)`: Creates a KMeans clustering model to partition the data into 2 clusters (`n_clusters=2`).
- `model.fit(data)`: Trains the model using the provided data.
- `label = model.predict(data)`: Predicts the cluster labels for each data point, assigning each point to one of the 2 clusters.
   - `label` will be an array where 0 represents the 1st cluster and 1 represents the 2nd cluster.

### 5. Visualizing Clusters

```python
plt.scatter(data['x'], data['y'], c=label)
plt.show()
```

- This visualizes the clustering results by color-coding the points based on their predicted cluster label (`c=label`).
- Points belonging to the same cluster are shown in the same color.

### 6. Calculate Silhouette Score

```python
from sklearn.metrics import silhouette_score
sc = silhouette_score(data, label, metric='euclidean')
print('Silhouette Coefficient: %.5f' % sc)
```

- `from sklearn.metrics import silhouette_score`: Imports the silhouette score function, which measures how well-separated the clusters are.
- `sc = silhouette_score(data, label, metric='euclidean')`: Calculates the silhouette score using the Euclidean distance metric. The silhouette score ranges from -1 to 1, where:
  - A score closer to 1 indicates good clustering (points are well separated).
  - A score closer to -1 suggests that the points might be wrongly clustered.
- `print('Silhouette Coefficient: %.5f' % sc)`: Prints the silhouette coefficient with five decimal places.

### Code Conclusion

The silhouette score (`sc`) is used to evaluate the quality of the clustering. A value closer to 1 means that the clustering is good, as the data points are well grouped together within their clusters and far apart from other clusters.
  