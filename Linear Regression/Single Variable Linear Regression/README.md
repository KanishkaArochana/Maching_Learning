# Single Variable Linear Regression in Machine Learning

## Overview
Linear Regression is a statistical method used for predictive modeling. In Single Variable Linear Regression, the goal is to model the relationship between a dependent variable \( y \) and an independent variable \( x \) using a straight line.

The formula for Single Variable Linear Regression is:

\[ y = mx + c \]

Where:
- \( m \) is the slope of the line
- \( c \) is the intercept
- \( x \) is the independent variable (input)
- \( y \) is the dependent variable (output)

## Libraries Used
1. **NumPy**
    ```python
    import numpy as np
    ```
    NumPy is used for numerical operations, such as working with arrays and performing mathematical operations like reshaping and array manipulations.

2. **Pandas**
    ```python
    import pandas as pd
    ```
    Pandas is used for data manipulation and analysis. It helps in importing datasets, converting them into DataFrames, and managing data in an efficient way.

3. **Matplotlib**
    ```python
    import matplotlib.pyplot as plt
    ```
    Matplotlib is a plotting library that is used to create visualizations like scatter plots and line plots, which help in understanding the relationship between the variables.

4. **Scikit-learn**
    ```python
    from sklearn.linear_model import LinearRegression
    ```
    Scikit-learn provides various machine learning algorithms, including Linear Regression, to help build and evaluate predictive models.

## Steps for Performing Single Variable Linear Regression
1. **Import and Save Dataset**
    ```python
    data = pd.read_csv('/content/drive/MyDrive/DataSets/Book_1.csv')
    ```
    Import the dataset using Pandas `read_csv()` function.

2. **Show Dataset**
    ```python
    data
    ```
    Display the dataset to understand its structure and the columns used.

3. **Plot the Data**
    ```python
    plt.scatter(data.videos, data.views, color='red')
    plt.xlabel('Number of Videos') 
    plt.ylabel('Total Views')
    plt.title('Videos vs Views')
    plt.show()
    ```
    Visualize the dataset using a scatter plot, where `videos` is the independent variable (x) and `views` is the dependent variable (y).

4. **Convert Data into NumPy Arrays**
    ```python
    x = np.array(data.videos.values)
    y = np.array(data.views.values)
    ```
    Convert the values of the columns `videos` and `views` into NumPy arrays.

5. **Reshape Data**
    ```python
    model.fit(x.reshape(-1,1), y)
    ```
    Scikit-learn's Linear Regression model requires a 2D array for the feature (x). Reshaping x ensures it has the correct shape for model training.

6. **Train the Model**
    ```python
    model = LinearRegression()
    model.fit(x.reshape(-1,1), y)
    ```
    Use the `fit()` method to train the Linear Regression model on the data.

7. **Predict Values**
    ```python
    new_x = np.array(45).reshape((-1,1))
    pred = model.predict(new_x)
    ```
    After training, you can use the `predict()` method to predict the dependent variable (views) for a given independent variable (number of videos).

8. **Create the Best Fit Line**
    ```python
    m, c = np.polyfit(x, y, 1)
    plt.plot(x, m*x + c, color='blue')
    ```
    The `polyfit()` function from NumPy computes the slope (m) and intercept (c) of the best fit line. The line is then plotted using `plt.plot()`.

9. **Prediction Function**
    ```python
    y_new = m*45 + c
    ```
    You can manually compute the predicted value using the equation of the best fit line.

## Special Methods
- **fit()**
  The `fit()` method is used to train the Linear Regression model by fitting the best line to the data.

- **predict()**
  The `predict()` method is used to predict the output (dependent variable) using the trained model, given new input values.

- **polyfit()**
  The `polyfit()` method from NumPy is used to compute the slope (m) and intercept (c) of the best fit line. It fits a polynomial of a given degree (1 for linear regression) to the data.
```

