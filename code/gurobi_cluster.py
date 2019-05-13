from __future__ import print_function
from gurobipy import *
import random


from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/ESS521project/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

maple29data = pd.read_csv(data_folder/'output/prep2_MAP_sub29.csv')
maple29data.columns
maple29data.index
NBR_ITEMS = 468
parcelid =  maple29data.Site_ID
sed = maple29data.SedRed
cost = maple29data.Cost
costcons = sum(cost)
costcons
clusts = maple29data.NEAR_FID
clusterids = list(maple29data.NEAR_FID.unique())
clusterids
clustersize = maple29data.Cluster_size
sedrank = maple29data.Sedrank


Groundset  = range(NBR_ITEMS)
Groundset
Subsets   = range(2)
Set = [list(sed), list(cost)]
Set


SetObjPriority = [2, 1]
SetObjWeight   = [1.0, -1.0]
model = Model('multiobj')
Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')
Elem

cost_cons = quicksum(Elem[k]*Set[1][k] for k in range(len(Elem)))
model.addConstr(Elem.sum() <= 10)
model.addConstr(cost_cons <= 1600000)
model.ModelSense = GRB.MAXIMIZE
model.setParam(GRB.Param.PoolSolutions, 100)


for i in range(len(Elem)):
    cluster_size = clustersize[i]
    topn = int(np.ceil(cluster_size/2))
    if  sedrank[i] > topn:
        Set[0][i] = 0

for k in Subsets:
    objn = sum(Elem[i]*Set[k][i] for i in range(len(Elem)))
    model.setObjectiveN(objn, k, SetObjPriority[k], SetObjWeight[k],
                            1.0 + k, 0.01, 'Set' + str(k))
objn
model.update
model.optimize()    
obj1 = model.getObjective(0)  
obj2 = model.getObjective(1)
sedsum = obj1.getValue()
costsum = obj2.getValue()

sedsum
costsum
var = model.getVars()
var[0].x 
selected = []
for i in range(len(var)):
    if var[i].x == 1:
        selected.append(var[i])
        print (var[i])

len(selected)      

print(obj1.getValue())

print(obj2.getValue())

plt.scatter(sedsum, costsum)





# Create initial model
model = Model("poolsearch")
 # Create dicts for tupledict.prod() function
sedCoefDict = dict(zip(Groundset, sed))
costCoefDict = dict(zip(Groundset, cost))

# Initialize decision variables for ground set:
# x[e] == 1 if element e is chosen
Elem = model.addVars(Groundset, vtype=GRB.BINARY, name='El')
 # Set objective function
model.ModelSense = GRB.MAXIMIZE
model.setObjective(Elem.prod(sedCoefDict))