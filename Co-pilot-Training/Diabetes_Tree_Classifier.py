# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the Diabetes Dataset
diabetes_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')

# Exploratory Data Analysis

# 1. Display the first 5 rows of the dataframe
print("First 5 rows of the dataframe:")
print(diabetes_df.head())

# 2. Display the number of rows and columns in the dataframe
print("\nNumber of rows and columns in the dataframe:")
print(diabetes_df.shape)

# 3. Display the data types of each column
print("\nData types of each column:")
print(diabetes_df.dtypes)

# 4. Display the number of missing values in each column
print("\nNumber of missing values in each column:")
print(diabetes_df.isnull().sum())

# 5. Display the number of unique values in each column
print("\nNumber of unique values in each column:")
print(diabetes_df.nunique())

# Split the Dataset into Training and Testing Sets
X = diabetes_df.drop('Outcome', axis=1)  # Features
y = diabetes_df['Outcome']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print("\nShapes of the resulting datasets:")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Train a Decision Tree Classifier

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f'\nAccuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Visualize the Decision Tree

# Set the size of the plot
plt.figure(figsize=(20,10))

# Plot the decision tree
plot_tree(clf, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)

# Display the plot
plt.show()