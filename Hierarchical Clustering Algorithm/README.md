
# Hierarchical Clustering Algorithm (Unsupervised)

## Overview
Hierarchical Clustering is an unsupervised machine learning algorithm used to group similar data points into clusters. Unlike K-means clustering, it does not require specifying the number of clusters beforehand. The algorithm creates a tree-like structure called a dendrogram to represent the hierarchy of clusters.

There are two main methods of hierarchical clustering:
- Agglomerative Clustering (Bottom-up approach)
- Divisive Clustering (Top-down approach)

## 1. Agglomerative Clustering

### What is Agglomerative Clustering?
Agglomerative Clustering is a bottom-up approach where each data point starts as its own cluster. The algorithm then merges the closest clusters iteratively until all data points are in one cluster or a predefined number of clusters are formed.

### Real-World Example
Consider a situation where you want to group cities based on their geographical proximity. Initially, each city is considered a separate cluster. Agglomerative clustering would iteratively merge the closest cities into clusters until only a few clusters remain, which could represent geographical regions.

### How it Works
1. Start with each data point as its own cluster.
2. Calculate the distances between all pairs of clusters.
3. Merge the two closest clusters.
4. Repeat the process until all points are grouped into a single cluster or the required number of clusters is achieved.

### Distance Metrics Used
- **Euclidean Distance**: Measures the straight-line distance between two points.
- **Manhattan Distance**: Measures the distance between two points along axes at right angles (grid-like distance).
- **Cosine Distance**: Measures the angle between two vectors, useful for high-dimensional data like text.

### Dendrogram
A dendrogram is a tree-like diagram that shows the arrangement of the clusters and their merging process. It helps visualize the hierarchical relationships between clusters.

## Code Example (Agglomerative Clustering)


### Step 1: Importing Libraries

```python
import pandas as pd
```

`pandas`: A powerful library for data manipulation and analysis. It's used here to create a DataFrame from a dictionary, which is a common way to handle datasets in Python.

### Step 2: Creating the Dataset

```python
data = pd.DataFrame(data = {'x': [0,1.1,1,2,2,4,5,5], 'y': [0,1.5,4,2,3,1,0,4]})
```

`pd.DataFrame()`: Converts a dictionary into a DataFrame. This dataset contains two columns, x and y, which represent the coordinates of points in a 2D space.

`data`: A DataFrame holding the sample points for clustering.

### Step 3: Scatter Plot Visualization

```python
import matplotlib.pyplot as plt
plt.scatter(data.x, data.y)
```

`matplotlib.pyplot.scatter()`: This function creates a scatter plot, where `data.x` is plotted on the x-axis, and `data.y` is plotted on the y-axis.

**Purpose**: This step visualizes the points in 2D space to help understand their distribution.

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7]
for index, value in enumerate(numbers):
    plt.annotate(value, (data.x[index], data.y[index]), size=14)
```

`plt.annotate()`: This function is used to add labels (numbers) to the points on the scatter plot. The `enumerate()` method is used to iterate over the points and label each one with an index.

**Purpose**: Adding labels helps identify each point on the scatter plot.

### Step 4: Dendrogram

```python
import scipy.cluster.hierarchy as sc
dendrogram = sc.dendrogram(sc.linkage(data, method='ward'))
```

`scipy.cluster.hierarchy.linkage()`: The linkage method calculates the hierarchical clustering using different methods like 'ward', 'single', 'complete', etc. It calculates the distance between clusters based on the data points.

`method='ward'`: The Ward method minimizes the variance within clusters when merging them.

`sc.dendrogram()`: This function creates a dendrogram plot, which is a tree-like diagram showing the arrangement of clusters. It visually represents the hierarchical relationship between the data points.

**Purpose**: The dendrogram is used to help determine the optimal number of clusters by visualizing the distances between clusters.

```python
plt.title('Dendrogram')  # Title for the dendrogram plot
plt.ylabel('Euclidean distances')  # Y-axis label for the dendrogram
plt.show()
```

`plt.title()` and `plt.ylabel()`: Add titles and labels to the plot for better understanding.

`plt.show()`: Displays the plot.

## Step 5: Training the Model (Agglomerative Clustering)

```python
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=2, linkage='ward')
pred = model.fit_predict(data)
```

`AgglomerativeClustering`: A method from `sklearn` used for hierarchical clustering. It's called agglomerative because it starts with each point as its own cluster and then merges them.

`n_clusters=2`: Specifies the number of clusters the algorithm should find. In this case, it’s set to 2.

`linkage='ward'`: Specifies the method for calculating the distance between clusters, similar to the earlier dendrogram.

`model.fit_predict(data)`: This method fits the model to the data and then assigns each point to a cluster. It returns the cluster labels for each point in the dataset.

### Step 6: Displaying the Predicted Cluster Labels

```python
pred
```

`pred`: The predicted cluster labels for each data point are displayed here. For example, `array([1, 1, 1, 1, 1, 0, 0, 0])` means that points 0 to 4 are in cluster 1, and points 5 to 7 are in cluster 0.

```python
data['cluster'] = pred
data
```

`data['cluster'] = pred`: Adds the predicted cluster labels as a new column in the DataFrame, making it easier to visualize which data points belong to which cluster.

### Step 7: Dividing Data into Clusters

```python
cluster1 = data[data['cluster'] == 0]
cluster2 = data[data['cluster'] == 1]
```

`data[data['cluster'] == 0]` and `data[data['cluster'] == 1]`: Filters the DataFrame into two separate clusters: one where the cluster label is 0 (Cluster 1) and one where the cluster label is 1 (Cluster 2).

### Step 8: Visualizing the Clusters

```python
plt.scatter(cluster1.x, cluster1.y, color='red')
plt.scatter(cluster2.x, cluster2.y, color='blue')
```

`plt.scatter()`: These lines plot the points of Cluster 1 and Cluster 2 on the scatter plot using different colors (red for Cluster 1, blue for Cluster 2).

**Purpose**: This step visualizes the results of the clustering algorithm, making it easy to see how the points are divided into two distinct groups.

- `cluster1.x` and `cluster1.y`: The x and y coordinates for the points in cluster 1.
- `cluster2.x` and `cluster2.y`: The x and y coordinates for the points in cluster 2.


## 2. Divisive Clustering

### What is Divisive Clustering?
Divisive Clustering is a top-down approach. It starts with all the data points in a single cluster and iteratively divides the clusters into smaller ones based on dissimilarity, continuing until each data point is in its own cluster or the desired number of clusters is achieved.

### Real-World Example
Imagine you are organizing a large group of people into smaller groups based on their interests. You start with everyone in a single group. Then, you split the group based on the most significant difference in interests (e.g., sports vs. art), and you continue to split the subgroups until each subgroup contains people with highly similar interests.

### How it Works
1. Start with all data points in a single cluster.
2. Divide the cluster based on the dissimilarity between data points.
3. Continue dividing the clusters until the required number of clusters is reached or until each data point forms its own cluster.

## Agglomerative vs. Divisive Clustering

| Feature | Agglomerative Clustering | Divisive Clustering |
|---------|---------------------------|---------------------|
| Approach | Bottom-up | Top-down |
| Start Point | Each point is its own cluster | All points are in one cluster |
| Process | Iteratively merges clusters | Iteratively divides clusters |
| Complexity | Faster for large datasets | More computationally expensive |
| Tree Representation | Dendrogram (hierarchical tree) | Dendrogram (hierarchical tree) |

## Key Takeaways
- Agglomerative Clustering is widely used due to its simplicity and effectiveness in grouping data.
- Divisive Clustering is more computationally expensive but can be useful for specific types of data.
- Hierarchical clustering does not require a predefined number of clusters and offers a flexible approach to discovering natural groupings in the data.

### Summary

- Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy of clusters.
- Dendrogram visualization is used to understand the relationships between clusters.
- `AgglomerativeClustering` from `sklearn` is used to perform the clustering and predict the clusters for each data point.
- The resulting clusters are visualized using a scatter plot, where each cluster is represented by a different color.
- This approach is useful when you need to understand the structure of your data and identify inherent patterns or groupings without prior knowledge of the number of clusters.

## Conclusion
Hierarchical Clustering is a powerful unsupervised learning algorithm for grouping data points into clusters based on similarity. Whether you use the Agglomerative or Divisive approach, hierarchical clustering provides a visual and intuitive understanding of the relationships between data points through dendrograms. This flexibility makes it an excellent choice for a wide range of applications.
