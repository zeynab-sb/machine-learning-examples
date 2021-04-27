from matplotlib import pyplot
from pandas import read_csv
import seaborn as sns
from pandas.plotting import scatter_matrix

path = r"breast-cancer.data"
headernames = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
data = read_csv(path, names=headernames)
print(data.head(50))

#size
print("************************* Shape *************************")
print(data.shape)
print("*********************************************************")

print("************************* Class Distribution *************************")
for header in headernames:
    count_class = data.groupby(header).size()
    print(count_class)
    print("------------------------------------------------------------------")
print("**********************************************************************")

print("************************* Type of features ***************************")
print(data.dtypes)
print("**********************************************************************")

print("******************* Correlation between features *********************")
correlations = data.corr(method='pearson')
print(correlations)
print("**********************************************************************")

print("***************************** Density ********************************")
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()
print("**********************************************************************")

print("***************************** Scatter ********************************")
scatter_matrix(data)
pyplot.show()
print("**********************************************************************")