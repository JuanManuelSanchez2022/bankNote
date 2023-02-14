import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import statsmodels.api as sm
import seaborn as sb

data = pd.read_csv('Banknote-authentication-dataset-.csv')

v1 = data['V1']
v2 = data['V2']
v1v2 = np.column_stack((v1, v2))
plt.scatter(v1,v2)
km_res = KMeans(n_clusters=3).fit(v1v2)

print (min(v1))
print (max(v1))
mean_v1 = sum(v1)/len(v1)
print(f"Mean v1: {mean_v1}")
print("Standard deviation: ", np.std(v1))


print (min(v2))
print (max(v2))
mean_v2 = sum(v2)/len(v2)
print(f"Mean v2: {mean_v2}")
print("Standard deviation: ", np.std(v2))


clusters = km_res.cluster_centers_
plt.scatter(v1,v2)
plt.xlabel('V1')
plt.ylabel('V2')

plt.scatter(clusters[:,0],clusters[:,1], s=1000, alpha=0.5, color='b')

#load data
df = pd.read_csv('Banknote-authentication-dataset-.csv')
#df_labels = pd.read_csv('Banknote-authentication-dataset-.csv')
print(df)

# set up to view all the info of the columns
2
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df.head()

df.shape

df.describe()

df.info()

df[df.duplicated()].shape[0]

sb.scatterplot(data=df, x='V1', y='V2')

sb.kdeplot(v1, v2, shade=True)

sb.set_style('whitegrid')

sb.kdeplot(v1, v2,
           shade=True,
          shade_lowest=False,
          cbar=True);

