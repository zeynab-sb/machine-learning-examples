from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt





path = r"breast-cancer.data"
headernames = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
data = read_csv(path, names=headernames)
print(data.head(50))

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

X = new_data.drop('Class', axis=1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = y_kmeans, s = 20, cmap = 'summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9);
plt.show()


