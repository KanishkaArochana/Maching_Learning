
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is an unsupervised machine learning algorithm used for clustering data. Unlike algorithms like K-Means, DBSCAN can identify clusters of any shape and can also detect outliers or noise. It groups data based on the density of points in a region and works well when clusters are of varying shapes and sizes.

## Comparison between K-Means, Hierarchical Clustering, and DBSCAN

| **Feature**                   | **K-Means**                                   | **Hierarchical Clustering**                      | **DBSCAN**                                    |
|-------------------------------|-----------------------------------------------|--------------------------------------------------|-----------------------------------------------|
| **Cluster Structure**          | Partitions data into a predefined number of clusters | Builds a tree-like structure (dendrogram)         | Based on density, groups points with high density into clusters |
| **Predefined Number of Clusters** | Yes, requires specifying the number of clusters (k) | No, clusters are defined based on the data's structure | No, clusters are determined based on density |
| **Cluster Shape Assumption**   | Assumes spherical clusters of equal size     | No specific assumption about cluster shape       | Can form clusters of arbitrary shapes |
| **Outlier Sensitivity**        | Sensitive to outliers (can affect centroids)  | Can be less sensitive to outliers depending on linkage method | Robust to outliers, can identify noise points |
| **Computational Complexity**   | Fast for large datasets (O(n))                | Computationally expensive for large datasets (O(n²) or O(n³)) | Efficient for large datasets (O(n log n)) |
| **Scalability**                | Scales well to large datasets                 | Struggles with very large datasets               | Scales well for large datasets |
| **Use Case**                   | Works best with spherical, well-separated clusters | Suitable for small datasets and hierarchical relationships | Works well with irregularly shaped clusters and noise detection |
---

## How DBSCAN Works

DBSCAN requires two main parameters:

**Eps (ε):**

- Eps is the radius (neighborhood) within which DBSCAN looks for nearby points to form a cluster.
- Example: If Eps = 2, it means DBSCAN will look for points within a radius of 2 units around a given point.

**MinPts:**

- MinPts is the minimum number of points that must be present within the Eps radius to form a dense region, or core point.
- Example: If MinPts = 3, at least 3 points must be within the Eps radius for a point to be considered a core point.

## Types of Points in DBSCAN

**Core Point:**

- A point that has at least MinPts number of neighboring points (within the Eps radius).
- Example: If Eps = 2 and MinPts = 3, a point is a core point if there are at least 3 other points within a distance of 2 units from it.

**Border Point:**

- A point that has fewer than MinPts points within the Eps radius, but is within the neighborhood of a core point.
- These points are on the border of a cluster and may be near other clusters.
- Example: A point is a border point if it has 2 points within the Eps radius, but it is within the Eps radius of a core point.

**Noise Point (Outlier):**

- A point that does not meet the criteria of either a core point or a border point. These points are considered noise and do not belong to any cluster.
- Example: If a point has no neighbors within the Eps radius and is not within the radius of any core point, it is considered noise.


## Example Parameters for DBSCAN

- **Eps = 2**: This means that a point can belong to the same cluster as another if the distance between them is less than or equal to 2.
- **MinPts = 3**: This means that for a point to be considered a core point, it must have at least 3 points (including itself) within a distance of 2.

### Explanation with Example

Consider the following set of points:
```
[[3,5], [3,4], [4,4], [9,6], [8,7], [10,6], [9,7], [5,10], [8.5,6], [2,3], [2.5,4]]
```

#### Step-by-Step Process

**Core Points:**

1. For each point, check how many other points are within a distance of Eps = 2.
2. If a point has at least MinPts = 3 neighbors, it is classified as a core point.
   - For example, point `[3, 5]` has neighbors `[3, 4]` and `[4, 4]`, which are within Eps = 2 distance, and since it has 3 points (including itself) within the distance, it's a core point.

**Border Points:**

- Border points are not core points but lie within Eps = 2 distance of a core point.
  - For example, point `[9, 7]` is within Eps = 2 distance of `[9, 6]` and `[8, 7]` (core points), so it becomes a border point.

**Noise Points:**

- Noise points do not belong to any cluster because they don't satisfy the conditions of being a core or border point.
  - For example, point `[2, 3]` might not have enough neighbors within Eps = 2 to be a core point, nor is it near any core points, so it's classified as a noise point.


## How to Identify Clusters with DBSCAN

1. **Calculate the Distance Matrix:** Calculate the distance between every pair of points in the dataset.
2. **Check if Points are Core Points:** For each point, count how many other points are within the Eps distance. If there are enough points (≥ MinPts), mark it as a core point.
3. **Expand Clusters:** Start with a core point and assign it to a cluster. Then, recursively add all reachable points (those within the Eps radius of the core point) to the cluster.
4. **Border Points and Noise:** Border points are added to the cluster if they are reachable from core points, but noise points remain unassigned.
5. **Resulting Clusters:** Once the algorithm has assigned all points, the clusters are formed, and the noise points are identified.

## Example
- Start with a random point. If it is a core point, all the points in its neighborhood (within Eps) are added to the cluster.  
- If a point is a border point, it is added to the cluster of the core point it is associated with.  
- Repeat this process for all points, forming clusters. Points that are not reachable from any core points are marked as noise.

### Example with Points:
Given:
```
[[3,5], [3,4], [4,4], [9,6], [8,7], [10,6], [9,7], [5,10], [8.5,6], [2,3], [2.5,4]]
```

- **Cluster 1**: Core points like `[3,5]`, `[3,4]`, and `[4,4]` form a cluster. This cluster would include `[3,5]`, `[3,4]`, `[4,4]`.
- **Cluster 2**: Core points `[9,6]`, `[8,7]`, and `[10,6]` form another cluster.
- **Noise**: Points like `[2,3]` might be noise since they don't 
belong to any core's neighborhood.


---

## Code Explanation

### Step 1: Import Libraries
We first import the necessary libraries for data manipulation, visualization, and machine learning.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
```

- `numpy` is for numerical operations.
- `pandas` is used to handle data in DataFrame format.
- `matplotlib.pyplot` is used for visualizing the data.
- `DBSCAN` is the algorithm from `sklearn` used to perform clustering.

### Step 2: Create a DataFrame (Data Set)
Here, we create a simple 2D dataset with 2 columns using a DataFrame.

```python
data = pd.DataFrame([[3, 5], [3, 4], [4, 4], [9, 6], [8, 7], [10, 6], [9, 7], [5, 10], [8.5, 6], [2, 3], [2.5, 4]])
```

This dataset consists of 11 data points, each with two numerical features.

### Step 3: Visualize the Data
To understand the data visually, we can plot the points on a scatter plot.

```python
plt.scatter(data[0], data[1])  # x --> 0 column, y --> 1 column
plt.show()
```

This generates a scatter plot of the data points. The x-axis corresponds to the first column of the DataFrame, and the y-axis corresponds to the second column.

### Step 4: Import DBSCAN 

```python
from sklearn.cluster import DBSCAN
```
- `DBSCAN` is the algorithm from `sklearn` used to perform clustering.

### Step 5: Apply DBSCAN Clustering
Now, we apply the DBSCAN clustering algorithm to the dataset.

```python
dbscan = DBSCAN(eps=2, min_samples=3)
```

- `eps=2` specifies the maximum distance between two samples for one to be considered as in the neighborhood of the other. It's the radius of the neighborhood.
- `min_samples=3` is the minimum number of samples required to form a dense region (i.e., a cluster).

Now, we fit the DBSCAN model to our data:

```python
dbscan.fit(data)
```

This will compute the clusters.

### Step 6: View the Cluster Labels
After the clustering is complete, we can check the labels assigned to each data point. Each point gets a label indicating its cluster.

```python
dbscan.labels_
```

This will output an array like this:

```scss
array([ 0,  0,  0,  1,  1,  1,  1, -1,  1,  0,  0])
```

- `0` and `1` represent two different clusters.
- Cluster `0` contains points 0, 1, 2, 8, 9, 10 (index starts at 0).
- Cluster 1 contains points 3, 4, 5, 6, 7.
- `-1` represents noise points, i.e., points that do not belong to any cluster.

### Step 7: Visualize the Clusters
Finally, we visualize the clustered data by coloring the points according to their cluster labels.

```python
plt.scatter(data[0], data[1], c=dbscan.labels_)
plt.show()
```

This will plot the data points, with different colors representing different clusters. Noise points (label `-1`) will be displayed with a unique color.

## Summary
- DBSCAN groups points that are closely packed together while marking sparse points as noise.
- The `eps` parameter controls the maximum distance between two points to be considered as neighbors.
- The `min_samples` parameter defines the minimum number of points required to form a cluster.
- By using `dbscan.labels_`, we can identify clusters and noise points.



