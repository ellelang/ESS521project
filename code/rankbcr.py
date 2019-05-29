from __future__ import print_function
import random
from pathlib import Path
data_folder = Path("C:/Users/langzx/Desktop/github/ESS521project/data")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
maple29data = pd.read_csv(data_folder/'output/prep2_MAP_sub29.csv')
maple29data.columns
maple29data.index
NBR_ITEMS = 468
top = np.arange(10,469,10)
top
range(len(top))
topname = ["Top" + str(i) for i in top]
topname


for i in range (len(top)):
    maple29data[topname[i]] = [0]* NBR_ITEMS


maple29data.iloc[maple29data.nlargest(3,'bcr').index]

maple29data.loc[maple29data.nlargest(3,'bcr').index,'Top10'] = 1 
maple29data.loc[maple29data.nlargest(3,'bcr').index]['Top10']

maple29data.columns

for t in range(len(top)):
    maple29data.loc[maple29data.nlargest(top[t],'bcr').index,topname[t]] = 1 

#maple29data.to_csv(data_folder/'output/bcr_ranking_MAPsub29.csv', index = False)

sed = maple29data['SedRed']
cost = maple29data['Cost']

sum(sed * maple29data['Top10'])

sedsum = [sum(sed * maple29data[i]) for i in topname]

costsum = [sum(cost * maple29data[i]) for i in topname]



#########add interactions
topname_epis = ["Top_epis" + str(i) for i in top]
topname_epis
cluster_size = maple29data['Cluster_size']
sedrank =  maple29data['Sedrank']



for i in range (len(top)):
    maple29data[topname_epis[i]] = [0]* NBR_ITEMS
    
bcr_epis = [None]*NBR_ITEMS
sed_epis = [None]*NBR_ITEMS

for i in range(NBR_ITEMS):
    topn = int(np.ceil(cluster_size[i]/2))
    if sedrank[i] > topn:
        bcr_epis[i] = 0
        sed_epis[i] = 0
    else:
        bcr_epis[i] = sed[i]/cost[i]
        sed_epis[i] = sed[i]

bcr_epis
sed_epis
maple29data['bcr_epis'] = bcr_epis
maple29data['bcr_epis']
maple29data['SedRed_epis'] = sed_epis

for t in range(len(top)):
    maple29data.loc[maple29data.nlargest(top[t],'bcr_epis').index,topname_epis[t]] = 1

maple29data.to_csv(data_folder/'output/bcr_ranking_MAPsub29.csv', index = False)


sedsum_epis = [sum(sed_epis * maple29data[i]) for i in topname_epis]
sedsum_epis 
costsum_epis = [sum(cost * maple29data[i]) for i in topname_epis]
costsum_epis

sedsum = [sum(sed_epis * maple29data[i]) for i in topname]

costsum = [sum(cost * maple29data[i]) for i in topname]



df_seeds = maple29data[topname]
df_seeds.to_csv(data_folder/'output/bcr_seeds.csv', index = False)

df_seeds_epis = maple29data[topname_epis]
df_seeds.to_csv(data_folder/'output/bcr_seeds_epis.csv', index = False)


seeds_array = np.array(maple29data[topname])
seeds_array
seeds_array_to_list = seeds_array.tolist()
json_file = "bcrseeds.json" 
json.dump(seeds_array_to_list, open(data_folder/json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)

seeds_array_epis = np.array(maple29data[topname_epis])
seeds_array_epis_to_list = seeds_array_epis.tolist()
json_file = "bcr_seeds_epis.json" 
json.dump(seeds_array_epis_to_list, open(data_folder/json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)


plt.scatter(sedsum, costsum, c='b', marker='x', label='bcr_ranking')
plt.scatter(sedsum_epis,costsum_epis, c = 'c', marker = 'o', label='bcr_epistasis')
plt.show()

plt.scatter(noseedsed_f, noseedcost_f,c='y', marker='v', label='EA_noseed')
plt.scatter(sed_f, cost_f, c='r', marker='s', label='EA_bcrseed')
plt.legend(loc='upper left')
plt.xlabel('Sed_Reduction')
plt.ylabel('Cost')
plt.savefig('maple29_paretofront_compare.png',dpi = 100)
plt.show()



plt.scatter(sedsum, costsum)

plt.scatter(sed_f, cost_f)
