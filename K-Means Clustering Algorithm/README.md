
# K-Means Clustering Algorithm in Machine Learning

## What is Data Clustering in Machine Learning?

Data clustering is a technique in machine learning where data points are grouped based on their similarity. It aims to identify patterns or structures in unlabeled data by organizing it into clusters, where data points within the same cluster are more similar to each other than to those in other clusters.

## What is the K-Means Clustering Algorithm?

K-Means is an unsupervised machine learning algorithm used for clustering. It partitions data into K clusters by minimizing the variance within each cluster and ensuring that each data point belongs to the nearest cluster centroid.

## Applications of Unsupervised Learning

- **Customer Segmentation**: Grouping customers based on purchasing behavior.
- **Search Engines**: Organizing search results into clusters for better user experience.
- **Recommendation Systems**: Grouping similar items or users for personalized recommendations.

## How K-Means Clustering Works (Step-by-Step with a Real-World Example)

### Initialize:

1. Choose the number of clusters K.
2. Randomly initialize K cluster centroids.

### Assign Data Points to Clusters:

1. For each data point, calculate the distance to each centroid.
2. Assign the data point to the nearest centroid.

### Update Centroids:

1. Compute the new centroid for each cluster as the mean of all points assigned to that cluster.

### Repeat:

1. Repeat steps 2 and 3 until the centroids stabilize (i.e., no significant change).

### Real-World Example:

Consider a retail store analyzing customers based on their annual income and spending score to create targeted marketing campaigns.

1. Choose K = 5 (five customer segments).
2. Assign each customer to the nearest centroid based on their income and spending score.
3. Update the centroids based on the mean of each cluster.
4. Iterate until centroids stabilize.

## How to Identify the Value of K?

The value of K determines the number of clusters. Choosing the right K is crucial for meaningful clustering.

### Elbow Method:

**WCSS (Within-Cluster Sum of Squares):** Measures the total variance within clusters.

#### Procedure:

1. Compute WCSS for different values of K.
2. Plot K values against WCSS.
3. Identify the "elbow point," where the WCSS curve has the steepest decline.
4. **Optimal K:** The value of K at the elbow point.

#### Example WCSS Errors for Different K:

```python
wcss_error = [269981.28, 185930.46, 106348.37, 73880.64, 44448.45]
```
## Distances in K-Means

- **Euclidean Distance**: The straight-line distance between two points.
- **Manhattan Distance**: The sum of the absolute differences between the coordinates of two points.


## Steps to Implement K-Means Clustering

1. **Import Data Sets**
    ```python
    import pandas as pd
    data = pd.read_csv('/content/drive/MyDrive/DataSets/Customers.csv')
    ```
    **Explanation:**
    We use the pandas library to read the CSV file containing customer data. The file path needs to be adjusted to where the dataset is stored. `data` now holds the dataset in the form of a DataFrame.

2. **Random Rows**
    ```python
    data.sample(5)
    ```
    **Explanation:**
    This line returns a random sample of 5 rows from the dataset to inspect a few data points and ensure the dataset is loaded correctly.

3. **Select Relevant Columns**
    ```python
    data = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    data.sample(5)
    ```
    **Explanation:**
    We select only the relevant columns for clustering: 'Annual Income (k$)' and 'Spending Score (1-100)'. `data.sample(5)` ensures the first 5 rows are displayed for review.

4. **Rename Columns**
    ```python
    data = data.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'})
    data.sample(5)
    ```
    **Explanation:**
    We rename the columns to make them more readable and simpler, changing 'Annual Income (k$)' to 'Income' and 'Spending Score (1-100)' to 'Score'. We use `data.sample(5)` to verify the column names after renaming.

5. **Visualizing the Data**
    ```python
    import matplotlib.pyplot as plt
    plt.scatter(data['Income'], data['Score'])  # x = Income, y = Score
    plt.xlabel('Income')
    plt.ylabel('Score')
    plt.show()
    ```
    **Explanation:**
    We import matplotlib for plotting. A scatter plot is created to visually inspect the relationship between 'Income' and 'Score'. Each point represents a customer. The x-axis corresponds to 'Income', and the y-axis corresponds to 'Score'.

6. **Finding the Optimal Number of Clusters (K)**
    ```python
    from sklearn.cluster import KMeans
    # List of potential K values
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # WCSS (Within-Cluster Sum of Squares) value storage
    wcss_error = []

    # Loop to compute WCSS for each K value
    for k in k_values:
        model = KMeans(n_clusters=k)
        model.fit(data[['Income', 'Score']])
        wcss_error.append(model.inertia_)
    ```
    **Explanation:**
    We use the KMeans model from sklearn.cluster and loop through a range of potential values for K (from 1 to 10). For each value of K, the model is trained using the 'Income' and 'Score' data, and the WCSS (Within-Cluster Sum of Squares) is calculated. This value measures the compactness of the clusters.

7. **Plotting WCSS for Optimal K**
    ```python
    # Plotting the WCSS error vs K values
    plt.plot(k_values, wcss_error)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS Error')
    plt.show()
    ```
    **Explanation:**
    A plot is generated to visualize how WCSS changes with the number of clusters (K). The "elbow" method helps in identifying the optimal value of K. The plot will show the point where the WCSS starts to decrease more slowly, suggesting the best K value.

8. **Train the K-Means Model**
    ```python
    model = KMeans(n_clusters=5)
    pred = model.fit_predict(data[['Income', 'Score']])
    ```
    **Explanation:**
    We create a KMeans model with K=5 (based on the previous plot). The model is then fitted to the data, and `fit_predict()` assigns each data point to one of the 5 clusters. The predicted cluster assignments are stored in `pred`.

9. **Viewing Predicted Cluster Labels**
    ```python
    pred
    ```
    **Explanation:**
    This prints out the predicted cluster labels for each data point, indicating which cluster each customer belongs to.

10. **Add Cluster Labels to Data**
    ```python
    data['Cluster'] = pred
    data.head()
    ```
    **Explanation:**
    We add the predicted cluster labels (stored in `pred`) as a new column 'Cluster' in the DataFrame `data`. `data.head()` displays the first few rows of the dataset with the newly added cluster labels.

11. **Separate Data by Clusters**
    ```python
    c1 = data[data['Cluster'] == 0]
    c2 = data[data['Cluster'] == 1]
    c3 = data[data['Cluster'] == 2]
    c4 = data[data['Cluster'] == 3]
    c5 = data[data['Cluster'] == 4]
    ```
    **Explanation:**
    The dataset is separated into individual DataFrames based on the cluster labels (0, 1, 2, 3, 4). This allows us to analyze each cluster separately.

12. **Plotting the Clusters**
    ```python
    plt.scatter(c1['Income'], c1['Score'], color='red')
    plt.scatter(c2['Income'], c2['Score'], color='blue')
    plt.scatter(c3['Income'], c3['Score'], color='green')
    plt.scatter(c4['Income'], c4['Score'], color='yellow')
    plt.scatter(c5['Income'], c5['Score'], color='black')

    # Plot the centers of the clusters
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color='purple')
    plt.xlabel('Income')
    plt.ylabel('Score')
    plt.show()
    ```
    **Explanation:**
    We plot the clusters, each with a different color (red, blue, green, yellow, and black) to visualize how the data points are grouped. The cluster centers are plotted using the `model.cluster_centers_` attribute, which gives the coordinates of the center of each cluster. These are marked with the color purple.

13. **View Cluster Centers**
    ```python
    model.cluster_centers_
    ```
    **Explanation:**
    This displays the coordinates of the center of each cluster. These are the values that represent the "mean" or centroid of each cluster.


## Conclusion

K-Means Clustering is a powerful algorithm for grouping data points into distinct clusters based on similarity. By following the steps outlined above, you can effectively implement K-Means in Python and gain insights from your dataset.
