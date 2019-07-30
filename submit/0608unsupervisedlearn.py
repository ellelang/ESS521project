from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
from sklearn.cluster import KMeans
wcmodata_mapsub29 = pd.read_csv(data_folder/'output/wcmo_MAP_sub29.csv')
mapsub29 = pd.read_csv(data_folder/'output/wcmo_MAP_sub29_unsupervise.csv', index_col = 'Site_ID')
mapsub29.head()
mapsub29.columns
mapsub29.shape

mapsub29['LAND_COVER'] = mapsub29['LAND_COVER'].astype('category')
mapsub29['FrmlndCls'] = mapsub29['FrmlndCls'].astype('category')
mapsub29['DrainClass'] = mapsub29['DrainClass'].astype('category')
mapsub29['HydrcRatng'] = mapsub29['HydrcRatng'].astype('category')
mapsub29['SurfText'] = mapsub29['SurfText'].astype('category')
mapsub29['PondFCls'] = mapsub29['PondFCls'].astype('category')
mapsub29['NEAR_FC'] = mapsub29['NEAR_FC'].astype('category')

cat_columns = mapsub29.select_dtypes(['category']).columns
mapsub29[cat_columns] = mapsub29[cat_columns].apply(lambda x: x.cat.codes)
mapsub29['NEAR_FC']
# Declaring Model


map29keys = mapsub29[['SedRed','Cost','Duck']]
inertiatest = []
ncluster = list(range(10,150,10))
ncluster

for c in ncluster:
    model = KMeans(n_clusters= c)
    modelc = model.fit(map29keys)
    inertiatest.append(modelc.inertia_)
    
plt.plot(ncluster, inertiatest,'.-')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.grid(None)
plt.savefig('maple29_Inertia.png',dpi = 300,bbox_inches='tight')
plt.show()


################
model1 = KMeans(n_clusters= 30)
map29keys.shape
# Fitting Model
model1 = model.fit(map29keys)
model1.inertia_

# Prediction on the entire data
sedred = map29keys['SedRed']
cost = map29keys['Cost']
duck = map29keys['Duck']
cost
labels = model1.predict(map29keys)
plt.scatter(sedred, cost, c = labels)
#plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.grid(None)
plt.savefig('maple29cluster_label.png', dpi=300)
print(labels)
labellist = labels.tolist()
mapsub29['Labels'] = labellist


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sedred, duck, cost, c=labels)
ax.set_xlabel('Sed_Reduction')
ax.set_ylabel('Duck')
ax.set_zlabel('$')
#ax.grid(False)
plt.savefig('maple29_3d.png',dpi = 300,bbox_inches='tight')

plt.show()
###### for Maple subbasin
mapledata = pd.read_csv(data_folder/'output/wcmo_MAP.csv')
maplekeys = mapledata[['SedRed','Cost','Duck']]
clusterids = list(mapledata.NEAR_FID.unique())
len(clusterids)

mapleinertiatest = []
maplencluster = list(range(50,547,50))
maplencluster

for c in maplencluster:
    model = KMeans(n_clusters= c)
    modelc = model.fit(maplekeys)
    mapleinertiatest.append(modelc.inertia_)

plt.plot(maplencluster, mapleinertiatest,'.-')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.grid(None)
plt.savefig('maplecluster_Inertia.png',dpi = 300,bbox_inches='tight')

model2 = KMeans(n_clusters= 150)
maplekeys.shape
# Fitting Model
model2 = model.fit(maplekeys)
model2.inertia_
sedred_maple = maplekeys['SedRed']
cost_maple = maplekeys['Cost']
duck_maple = maplekeys['Duck']
labels_maple = model2.predict(maplekeys)
len(labels_maple)
len(sedred_maple)
plt.scatter(sedred_maple, cost_maple, c = labels_maple)
plt.grid(None)
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.savefig('maplecluster_LABEL.png',dpi = 300,bbox_inches='tight')


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sedred_maple, duck_maple, cost_maple, c=labels_maple)
ax.set_xlabel('Sed_Reduction')
ax.set_ylabel('Duck')
ax.set_zlabel('$')
#ax.grid(False)
plt.savefig('maplewhole_3d.png',dpi = 300,bbox_inches='tight')

plt.show()



################## for the whole watershed
wcmodata = pd.read_csv(data_folder/'output/wcmogawhole.csv')
wcmokeys = wcmodata[['SedRed','Cost','Duck']]
wcmoinertiatest = []
wcmoncluster = list(range(200,1935,100))
wcmoncluster
wcmoinertiatest
for c in wcmoncluster:
    model = KMeans(n_clusters= c)
    modelc = model.fit(wcmokeys)
    wcmoinertiatest.append(modelc.inertia_)

plt.plot(wcmoinertiatest,'.-')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.grid(None)
plt.savefig('wcmocluster_Inertia.png',dpi = 300,bbox_inches='tight')

