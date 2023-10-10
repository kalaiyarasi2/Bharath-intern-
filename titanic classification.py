#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


  # You need to have a CSV file with your dataset


# In[4]:


data = pd.read_csv(r'D:\bharathintern\Titanic-Dataset.csv')


# In[5]:


data


# In[6]:


# Data preprocessing
# You might need to handle missing data, encode categorical variables, and select relevant features
# For simplicity, we'll just select a few features for this example
selected_features = ['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']
X = data[selected_features]
y = data['Survived']


# In[7]:


# Handle missing values (replace with the mean for simplicity)
X['Age'].fillna(X['Age'].mean(), inplace=True)

# Encode categorical variables (Sex and Embarked)
X = pd.get_dummies(X, columns=['Sex', 'Embarked'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[8]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions for a new passenger (example)
new_passenger = pd.DataFrame([[3, 30, 1, 10, 0, 0, 1, 0, 0, 1]], columns=X.columns)
prediction = model.predict(new_passenger)
if prediction[0] == 39:
    print('This passenger is likely to survive.')
else:
    print('This passenger is likely not to survive.')

