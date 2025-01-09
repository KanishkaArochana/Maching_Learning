# Data Visualization with Matplotlib in Machine Learning
## 1. Bar Chart
A bar chart is great for comparing categorical data.

```python
# Bar Chart Example
categories = ['Category A', 'Category B', 'Category C']
values = [10, 20, 15]

plt.bar(categories, values, color=['blue', 'green', 'red'])
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

**What it does:**
- `plt.bar()` creates a vertical bar chart.
- `categories` specifies the labels for each bar.
- `values` represents the heights of the bars.
- You can customize colors (`color=['blue', 'green', 'red']`).
- The title and axis labels make the chart easier to understand.

## 2. Pie Chart
A pie chart shows proportions of a whole, useful for visualizing parts of a dataset.

```python
# Pie Chart Example
sizes = [25, 35, 40]
labels = ['Part A', 'Part B', 'Part C']
colors = ['gold', 'lightblue', 'lightgreen']
explode = [0, 0.1, 0]  # Highlight Part B

plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%')
plt.title('Pie Chart Example')
plt.show()
```

**What it does:**
- `plt.pie()` creates the pie chart.
- `sizes` defines the proportions.
- `labels` gives a name to each section.
- `colors` customizes the chart's appearance.
- `explode` separates specific slices (e.g., Part B).
- `autopct='%1.1f%%'` adds percentages to the slices.

## 3. Histogram
Histograms are useful for showing the distribution of numerical data.

```python
# Histogram Example
data = np.random.normal(0, 1, 1000)  # Generate random data

plt.hist(data, bins=20, color='purple', edgecolor='black')
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

**What it does:**
- `plt.hist()` creates the histogram.
- `data` contains random values generated using numpy.
- `bins=20` divides the data into intervals (or bins).
- The color and edge styles make the plot visually appealing.

## 4. Scatter Plot
Scatter plots are perfect for showing relationships between two variables.

```python
# Scatter Plot Example
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color='orange', marker='o')
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

**What it does:**
- `plt.scatter()` plots individual data points.
- `x` and `y` are arrays of random numbers between 0 and 1.
- `color='orange'` changes the points' color, and `marker='o'` sets their shape.

## 5. Format Strings in Line Plots
Format strings help style line plots with shortcuts.

```python
# Line Plot with Format Strings
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, 'r--')  # Red dashed line
plt.title('Line Plot with Format Strings')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

**What it does:**
- `plt.plot()` creates a line plot.
- `x` contains evenly spaced numbers, and `y` calculates their sine values.
- `'r--'` formats the line as dashed red.
- You can use other styles, like `'b-'` (blue solid line) or `'g.'` (green dots).

## 6. Label Parameters and Legend
Adding legends and labels enhances plot readability.

```python
# Multiple Line Plot with Legend
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='Sine Wave', color='blue')
plt.plot(x, y2, label='Cosine Wave', color='green')
plt.title('Line Plot with Legend')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()  # Add legend
plt.show()
```

**What it does:**
- Multiple lines are plotted for sine and cosine.
- `label` specifies the name for each line, shown in the legend.
- `plt.legend()` displays the legend box.

## Summary
Each plot showcases different functionalities of Matplotlib:

- **Bar Chart:** Compare categories.
- **Pie Chart:** Visualize proportions.
- **Histogram:** Analyze data distribution.
- **Scatter Plot:** Display relationships.
- **Line Plot with Formats:** Customize line styles.
- **Legend:** Differentiate multiple data sets.
```