
# Instance Based and Model Based Learning in Machine Learning

Machine learning models can be divided into two main categories based on their approach to learning:

## Instance Based Learning (Lazy Learning)

### Instance Based Learning (Lazy Learning)

Instance Based Learning, also known as Lazy Learning, focuses on storing the training data and using it directly during prediction. Unlike Model Based Learning, it does not create an explicit model during the training phase. Instead, it delays the learning process until it receives a query for prediction.

#### How Instance Based Learning Works

- It stores the training data in memory.
- Predictions are made by analyzing the stored data to find patterns or similarities relative to the input.
- Computational effort is pushed to the prediction phase.

#### Examples of Instance Based Learning Algorithms

- **K-Nearest Neighbors (KNN)**: Finds the closest training instances to the input and predicts based on their labels.
- **Locally Weighted Regression**: Uses a subset of data points close to the input and applies weighted regression.

## Model Based Learning (Eager Learning)

### Model Based Learning (Eager Learning)

Model Based Learning, or Eager Learning, involves building a model during the training phase. The model identifies patterns and relationships in the data, which are then used to make predictions.

#### How Model Based Learning Works

- It trains a model by analyzing the entire dataset.
- The model captures the underlying patterns and structures of the data.
- Predictions are made by applying the learned model to new inputs.

#### Examples of Model Based Learning Algorithms

- **Linear Regression**: Establishes a linear relationship between inputs and outputs.
- **Logistic Regression**: Predicts categorical outcomes based on input features.
- **Decision Tree**: Creates a tree structure where each node represents a decision rule.
- **Support Vector Machine (SVM)**: Finds a hyperplane to classify data points.

## Differences Between Instance Based and Model Based Learning

| Feature | Instance Based Learning | Model Based Learning |
|---------|-------------------------|----------------------|
| **Learning Approach** | Lazy (delayed until prediction) | Eager (done during training) |
| **Data Storage** | Stores entire training data | Builds a generalized model |
| **Training Phase** | Minimal or no training effort | Computationally intensive |
| **Prediction Phase** | Computationally expensive | Fast and efficient |
| **Examples** | KNN, Locally Weighted Regression | Linear Regression, SVM |
| **Use Cases** | Small datasets, local analysis | Large datasets, global analysis |
