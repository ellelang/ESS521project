from __future__ import print_function
import random
from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/ESS521project/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

maple29dataset = pd.read_csv(data_folder/'output/wcmo_MAP_sub29.csv')
maple29dataset.columns
datarf = maple29dataset[['Site_ID','SedRed','Cost','area_m2','DA_9_SA_m2','NEAR_DIST',
                         'mean_slope','mean_D_m','mean_V_m3','CropProdIn']]

datarf['bcr'] = maple29dataset.SedRed/maple29dataset.Cost
datarf.shape

X = datarf.iloc[:, 3:10]
X.columns
y = datarf.iloc[:, 10]
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0) 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators= 35, random_state=1)  
rf.fit(X_train, y_train)  
y_pred = rf.predict(X_test)   

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

importance_rf = pd.Series(rf.feature_importances_, index = X.columns)
sorted_importance = importance_rf.sort_values()
sorted_importance.plot (kind = 'barh',color = 'wheat')
plt.grid(None)
plt.ylabel('feature_importance')
plt.savefig('maple29rf.png',dpi = 300,bbox_inches='tight')

plt.show()

#####################wcmo

mapledataset = pd.read_csv(data_folder/'output/wcmo_MAP.csv')
mapledatarf = mapledataset[['Site_ID','SedRed','Cost','area_m2','DA_9_SA_m2','NEAR_DIST',
                         'mean_slope','mean_D_m','mean_V_m3','CropProdIn']]

mapledatarf['bcr'] = mapledataset.SedRed/mapledataset.Cost
mapledatarf.shape

mapleX = mapledatarf.iloc[:, 3:10]
mapleX.columns
mapley = mapledatarf.iloc[:, 10]
mapley
X_train, X_test, y_train, y_test = train_test_split(
        mapleX.values, mapley.values, test_size=0.2, random_state=0) 

maplerf = RandomForestRegressor(n_estimators= 150, random_state=1)  
maplerf.fit(X_train, y_train)  
y_pred = maplerf.predict(X_test)   

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

maple_importance_rf = pd.Series(maplerf.feature_importances_, index = mapleX.columns)
maple_sorted_importance = maple_importance_rf.sort_values()
maple_sorted_importance.plot (kind = 'barh',color = 'wheat')
plt.ylabel('feature_importance')
plt.savefig('maplewholerf.png',dpi = 300,bbox_inches='tight')
plt.grid(None)
plt.show()

###########

