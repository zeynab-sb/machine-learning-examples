from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures




path = r"breast-cancer.data"
headernames = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
data = read_csv(path, names=headernames)
print(data.head(50))

def convert(data):
    number = preprocessing.LabelEncoder()
    data['Class'] = number.fit_transform(data.Class)
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

X = new_data.drop('Class', axis=1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_pred)


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

print(X_poly)
