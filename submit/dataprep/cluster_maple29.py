from __future__ import (absolute_import, division, print_function)
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import numpy as np
import pandas as pd
near_wld = pd.read_csv(data_folder/'output/nearcluster_MAP_sub29.csv')
near_wld.head()
near_wld.index
near_wld.columns
clid = list(near_wld['NEAR_FID'])
clid

def countX(lst, x): 
    return lst.count(x)

lst = clid

near_wld['Cluster_size'] = [countX(lst, x) for x in clid]
near_wld['Cluster_size']


near_wld["Sedrank"] = near_wld.groupby("NEAR_FID")["SedRed"].rank("dense", ascending=False)
near_wld["Sedrank"]
near_wld["bcr"] = near_wld["SedRed"]/near_wld['Cost']
near_wld.to_csv(data_folder/'output/prep2_MAP_sub29.csv', index = False)
near_wld['NEAR_FID'].astype('category')
