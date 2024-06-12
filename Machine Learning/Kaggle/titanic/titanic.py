import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
df_targets = pd.read_csv('dataset/gender_submission.csv')


#Age, 
# Contar los valores nulos en una columna específica
print(df_train['Ticket'].nunique())












'''

# Data preprocessing

# We are going to use the following features for the input: Pclass, Sex, Age, Fare

inputs_train = df_train[['Pclass', 'Sex', 'Fare']]
targets_train = df_train['Survived']
inputs_test = df_test[['Pclass', 'Sex', 'Fare']]


# Transform categorical attribute and fill null values
#inputs_train['Sex'] = inputs_train['Sex'].map({'male': 0, 'female': 1})
inputs_train['Fare'] = inputs_train['Fare'].fillna(inputs_train['Fare'].mean())
inputs_test['Sex'] = inputs_test['Sex'].map({'male': 0, 'female': 1})
inputs_test[['Fare', 'Pclass']] = inputs_test[['Fare', 'Pclass']].fillna(inputs_test[['Fare', 'Pclass']].mean())

# Normalize train and test inputs
scaler = MinMaxScaler()
inputs_train[['Fare', 'Pclass']] = scaler.fit_transform(inputs_train[['Fare', 'Pclass']])
inputs_test[['Fare', 'Pclass']] = scaler.transform(inputs_test[['Fare', 'Pclass']])


# First we are gonna try with a RandomForestClassifier:

model = RandomForestClassifier()
scores = cross_val_score(model, inputs_train, targets_train, cv=100, scoring='accuracy')

# Imprimir los resultados
print(f"Scores de cada pliegue: {scores}")
print(f"Score medio: {np.mean(scores)}")
print(f"Desviación estándar del score: {np.std(scores)}")


'''