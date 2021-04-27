from pandas import read_csv
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

print("******************* Correlation between features *********************")
correlations = data.corr(method='pearson')
print(correlations)
print("**********************************************************************")

