
# Save and Load Machine Learning Models

This document provides an overview of two methods to save and load machine learning models: Pickle and Joblib. Both methods allow you to save trained models and reuse them without retraining, which saves time and computational resources.

## 1. Pickle

Pickle is a Python library used for serializing and deserializing objects. It is a versatile and commonly used method to save and load machine learning models.

### Features of Pickle:

- **Serialization**: Converts Python objects (like trained models) into a byte stream that can be stored in a file.
- **Deserialization**: Reloads the byte stream from a file and converts it back into the original Python object.
- **Versatility**: Can handle a wide range of Python objects, not limited to machine learning models.

### Advantages of Pickle:

- **Ease of Use**: Integrated into Python's standard library, making it readily accessible.
- **Wide Compatibility**: Supports various Python objects, making it flexible for different use cases.
- **Customizability**: Allows customization of serialization and deserialization processes using custom Pickle protocols.

### Limitations of Pickle:

- **Security**: Loading Pickle files can be insecure if the source is untrusted, as it may execute arbitrary code.
- **Performance**: Can be slower and consume more memory compared to Joblib for large datasets or complex models.

## 2. Joblib

Joblib is a Python library optimized for serializing and deserializing objects, especially large data structures and numerical arrays. It is particularly suited for machine learning models and workflows.

### Features of Joblib:

- **Optimized for Arrays**: Designed to efficiently handle NumPy arrays and other large numerical data.
- **Parallel Computing Support**: Enables efficient use of computational resources for parallelizable tasks.
- **File Compression**: Supports file compression to reduce storage requirements.

### Advantages of Joblib:

- **Efficiency**: Faster serialization and deserialization for models containing large numerical data.
- **Optimized Storage**: Produces smaller file sizes due to its compression capabilities.
- **Simplicity**: Focused on handling large data, making it ideal for machine learning workflows.

### Limitations of Joblib:

- **Limited Flexibility**: Primarily focused on numerical arrays and may not be as versatile as Pickle for handling non-numerical objects.
- **Dependency**: Requires installing the Joblib library separately, as it is not part of Python's standard library.


## Explain in code for Save and Load Models 

### 1. Get Sample Model

```python
from sklearn.datasets import load_iris
dataset = load_iris()
```

**Purpose:** Loads the Iris dataset, a commonly used dataset in machine learning for classification.  
**Dataset Content:** Includes data (features) and their corresponding labels (targets).

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
```

**Purpose:** Splits the dataset into training and testing sets.  
- `x_train` and `y_train` are used for training the model.  
- `x_test` and `y_test` are used for testing the model's accuracy.  
- `test_size=0.2`: Reserves 20% of the data for testing.

### 2. Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
```

**Purpose:** Initializes a Random Forest Classifier with 100 decision trees (`n_estimators=100`).

```python
rf.fit(x_train, y_train)
```

**Purpose:** Trains the Random Forest model using the training data.

```python
print("Accuracy : {}".format(rf.score(x_test, y_test)))
```

**Purpose:** Calculates and prints the accuracy of the model on the test dataset.

### 3. Target Dataset

```python
dataset.target
```

**Purpose:** Displays the target labels (e.g., species of the Iris flowers).

```python
dataset.target_names
```

**Purpose:** Displays the target names (e.g., `['setosa', 'versicolor', 'virginica']`).

---

## Save and Load Model Using Pickle

### 4. Import Pickle Library

```python
import pickle
```

**Purpose:** Import the pickle library for saving and loading models.

### 5. Save Model Using Pickle

```python
with open('rf_model1.pickle', 'wb') as file:
    pickle.dump(rf, file)
```

**Purpose:** Saves the trained Random Forest model to a file named `rf_model1.pickle`.  
**Explanation:**  
- `'wb'`: Opens the file in write-binary mode.  
- `pickle.dump(rf, file)`: Writes the model into the file.

### 6. Load Model Using Pickle

```python
with open("rf_model1.pickle", "rb") as file:
    model1 = pickle.load(file)
```

**Purpose:** Loads the saved model from `rf_model1.pickle`.  
**Explanation:**  
- `'rb'`: Opens the file in read-binary mode.  
- `pickle.load(file)`: Reads the model from the file.

### 7. Take Prediction

```python
SepalLength = 6.3
SepalWidth = 2.3
PetalLength = 4.6
PetalWidth = 1.3

model1.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
```

**Purpose:** Uses the loaded model (`model1`) to predict the class of a new data point based on the provided features.

---

## Save and Load Model Using Joblib

### 8. Import Joblib

```python
import joblib
```

**Purpose:** Import the Joblib library for saving and loading models.

### 9. Save Model Using Joblib

```python
joblib.dump(rf, 'rf_model2.joblib')
```

**Purpose:** Saves the trained Random Forest model to a file named `rf_model2.joblib`.  
**Alternative:** Can also use `.pickle` as the file extension.

### 10. Load Model Using Joblib

```python
model2 = joblib.load('rf_model2.joblib')
```

**Purpose:** Loads the saved model from `rf_model2.joblib`.

### 11. Take Prediction Using Joblib Model

```python
SepalLength = 6.3
SepalWidth = 2.3
PetalLength = 4.6
PetalWidth = 1.3

model2.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
```

**Purpose:** Uses the loaded model (`model2`) to predict the class of a new data point.


## Choosing Between Pickle and Joblib

| Feature         | Pickle                                | Joblib                                      |
|-----------------|---------------------------------------|---------------------------------------------|
| Performance     | Slower for large numerical data       | Faster and optimized for large data         |
| File Size       | Larger files without compression      | Smaller files with compression              |
| Flexibility     | Supports diverse Python objects       | Best suited for numerical arrays            |
| Ease of Use     | Included in Python standard library   | Requires additional installation            |
| Security        | Vulnerable to untrusted sources       | Similar security concerns                   |

Choose Pickle for general-purpose serialization and tasks involving diverse Python objects. Use Joblib for machine learning workflows with large numerical datasets to maximize efficiency.
