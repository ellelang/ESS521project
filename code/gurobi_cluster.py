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
clusts = maple29data.NEAR_FID
clusterids = list(maple29data.NEAR_FID.unique())
clusterids

Groundset    = range(NBR_ITEMS)
Groundset
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