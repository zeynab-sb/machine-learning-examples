from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score






path = r"breast-cancer.data"
headernames = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
data = read_csv(path, names=headernames)

def convert(data):
    number = preprocessing.LabelEncoder()
    data['age'] = number.fit_transform(data.age)
    data['menopause'] = number.fit_transform(data.menopause)
    data['tumor-size'] = number.fit_transform(data['tumor-size'])
    data['inv-nodes'] = number.fit_transform(data['inv-nodes'])
    data['node-caps'] = number.fit_transform(data['node-caps'])
    data['breast'] = number.fit_transform(data['breast'])
    data['breast-quad'] = number.fit_transform(data['breast-quad'])
    data['irradiat'] = number.fit_transform(data.irradiat)
    data=data.fillna(-999)
    return data

new_data = convert(data)

#DecisionTreeClassifier
X = new_data.drop('Class', axis=1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#prediction
y_pred = classifier.predict(X_test)
print("****************** Prediction ******************")
print(y_pred)
print("************************************************")

print("*************** Confusion Matrix ***************")
result = confusion_matrix(y_test, y_pred)
print(result)
print("************************************************")
