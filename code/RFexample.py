from __future__ import print_function
import random
from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/ESS521project/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv(data_folder/'example/petrol_consumption.csv')  
dataset.head()  
dataset.shape
X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]
X.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)  
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

importance_rf = pd.Series(regressor.feature_importances_, index = X.columns)
sorted_importance = importance_rf.sort_values()
sorted_importance.plot (kind = 'barh',color = 'wheat')
plt.show()
