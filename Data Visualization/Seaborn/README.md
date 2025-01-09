# Data Visualization with Seaborn in Machine Learning
## Wide Range of Plot Types
Seaborn supports various types of plots to explore data. Here are a few key examples:
###  Example 1. Line Plot
A line plot is used to visualize data trends over a continuous variable (e.g., time). In Seaborn, the method used for this is `sns.lineplot()`. It plots data points and connects them with a line.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [100, 120, 180, 150, 200]
})

# Line Plot
sns.lineplot(data=data, x='Month', y='Sales')

# Title
plt.title('Monthly Sales Trend')

# Display the plot
plt.show()
```
- `sns.lineplot(data=data, x='Month', y='Sales')`: Creates a line plot using the DataFrame data. The x axis represents the months, and the y axis represents the sales.
- `plt.title('Monthly Sales Trend')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

###  Example  2. Scatter Plot
A scatter plot visualizes the relationship between two continuous variables by showing points for each data pair. In Seaborn, `sns.scatterplot()` is used to create scatter plots.

### Code Explanation:
```python
from sklearn.datasets import make_regression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# Convert to DataFrame
scatter_data = pd.DataFrame({'Feature': X.flatten(), 'Target': y})

# Scatter Plot
sns.scatterplot(data=scatter_data, x='Feature', y='Target')

# Title
plt.title('Scatter Plot of Feature vs Target')

# Display the plot
plt.show()
```
- `make_regression()`: Generates a simple linear regression dataset.
- `sns.scatterplot(data=scatter_data, x='Feature', y='Target')`: Creates a scatter plot for the feature (x) and target (y) values.
- `plt.title('Scatter Plot of Feature vs Target')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

### Example 3. Histogram
A histogram is used to show the distribution of a single variable. It groups the data into bins and shows how many data points fall into each bin. In Seaborn, `sns.histplot()` is used to create histograms.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.randn(1000)

# Histogram
sns.histplot(data, bins=30, kde=True)

# Title
plt.title('Data Distribution')

# Display the plot
plt.show()
```
- `np.random.randn(1000)`: Generates 1000 random data points from a normal distribution.
- `sns.histplot(data, bins=30, kde=True)`: Creates a histogram with 30 bins. The `kde=True` argument adds a Kernel Density Estimate (KDE) plot to the histogram for a smoothed distribution curve.
- `plt.title('Data Distribution')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

## Statistical Enhancements
Seaborn provides advanced tools to visualize statistical data.

### Example 1. Regression Plot
A regression plot helps visualize the relationship between two variables by fitting a regression line. It also displays confidence intervals. In Seaborn, `sns.regplot()` is used for regression plots.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
scatter_data = pd.DataFrame({'Feature': [1, 2, 3, 4, 5], 'Target': [2, 4, 5, 4, 5]})

# Regression Plot
sns.regplot(data=scatter_data, x='Feature', y='Target', ci=None)

# Title
plt.title('Regression Plot')

# Display the plot
plt.show()
```
- `sns.regplot(data=scatter_data, x='Feature', y='Target', ci=None)`: Creates a regression plot with Feature on the x-axis and Target on the y-axis. The `ci=None` argument removes the confidence interval shading.
- `plt.title('Regression Plot')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

### Example 2:  Kernel Density Estimation (KDE) Plot
KDE plots are used to estimate the probability density function of a continuous variable. In Seaborn, `sns.kdeplot()` creates KDE plots.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.randn(1000)

# KDE Plot
sns.kdeplot(data, fill=True)

# Title
plt.title('KDE Plot')

# Display the plot
plt.show()
```
- `sns.kdeplot(data, fill=True)`: Creates a KDE plot for the data. The `fill=True` argument fills the area under the density curve.
- `plt.title('KDE Plot')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

## Categorical Data Visualization
Seaborn is excellent for visualizing categorical data.
### Example 1: Box Plot
Box plots are useful for visualizing the distribution of data across categories. They show the median, quartiles, and potential outliers. In Seaborn, `sns.boxplot()` creates box plots.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load Seaborn's built-in tips dataset
tips = sns.load_dataset('tips')

## Categorical Data Visualization
Seaborn is great for visualizing categorical data.
### Example 1:  Box Plot
sns.boxplot(data=tips, x='day', y='total_bill', palette='Set3')

# Title
plt.title('Box Plot of Total Bill by Day')

# Display the plot
plt.show()
```
- `sns.load_dataset('tips')`: Loads a built-in Seaborn dataset.
- `sns.boxplot(data=tips, x='day', y='total_bill', palette='Set3')`: Creates a box plot showing the distribution of total bills by day.
- `plt.title('Box Plot of Total Bill by Day')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

### Example 2:  Bar Plot
Bar plots are used for comparing quantities across categories. In Seaborn, `sns.barplot()` is used to create bar plots.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load Seaborn's built-in tips dataset
tips = sns.load_dataset('tips')

# Bar Plot
sns.barplot(data=tips, x='day', y='total_bill', ci='sd', palette='Set2')

# Title
plt.title('Bar Plot of Average Total Bill by Day')

# Display the plot
plt.show()
```
- `sns.barplot(data=tips, x='day', y='total_bill', ci='sd', palette='Set2')`: Creates a bar plot of the average total bill by day with standard deviation error bars (`ci='sd'`).
- `plt.title('Bar Plot of Average Total Bill by Day')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

### Example 3:  Violin Plot
Violin plots combine aspects of both box plots and KDEs, showing the distribution of the data across categories. In Seaborn, `sns.violinplot()` creates violin plots.

### Code Explanation:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load Seaborn's built-in tips dataset
tips = sns.load_dataset('tips')

# Violin Plot
sns.violinplot(data=tips, x='day', y='total_bill', palette='muted')

# Title
plt.title('Violin Plot of Total Bill by Day')

# Display the plot
plt.show()
```
- `sns.violinplot(data=tips, x='day', y='total_bill', palette='muted')`: Creates a violin plot of the total bill across days.
- `plt.title('Violin Plot of Total Bill by Day')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.


## Customization and Theming

Seaborn provides extensive customization options for styling your plots, such as changing themes and color palettes.

### Example 1: Changing Themes

Seaborn offers several themes (e.g., darkgrid, whitegrid, dark, white, and ticks) that affect the overall appearance of the plots. You can use `sns.set_theme()` to set the theme.

### Code Explanation:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [100, 120, 180, 150, 200]
})

# Setting a theme
sns.set_theme(style="darkgrid")

# Line Plot
sns.lineplot(data=data, x='Month', y='Sales')

# Title
plt.title('Line Plot with Darkgrid Theme')

# Display the plot
plt.show()
```

- `sns.set_theme(style="darkgrid")`: This sets the plot's style to darkgrid, which adds a grid with a dark background.
- `sns.lineplot(data=data, x='Month', y='Sales')`: Creates a line plot for the given data with months on the x-axis and sales on the y-axis.
- `plt.title('Line Plot with Darkgrid Theme')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.

### Example 2: Custom Color Palettes

Seaborn allows you to create custom color palettes. You can define color palettes using names like coolwarm, Blues, or create your own.

### Code Explanation:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load Seaborn's built-in tips dataset
tips = sns.load_dataset('tips')

# Custom Palette
palette = sns.color_palette("coolwarm", as_cmap=True)

# Heatmap
sns.heatmap(tips.corr(), annot=True, cmap=palette)

# Title
plt.title('Heatmap with Custom Palette')

# Display the plot
plt.show()
```

- `sns.color_palette("coolwarm", as_cmap=True)`: Creates a custom color palette using the coolwarm scheme, which is used to map the heatmap colors.
- `sns.heatmap(tips.corr(), annot=True, cmap=palette)`: Creates a heatmap of the correlations in the tips dataset, with annotations showing correlation values and applying the custom color palette.
- `plt.title('Heatmap with Custom Palette')`: Adds a title to the plot.
- `plt.show()`: Displays the plot.




