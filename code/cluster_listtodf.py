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
near_wld.to_csv(data_folder/'output/prep2_MAP_sub29.csv', index = False)
near_wld['NEAR_FID'].astype('category')

near_wld[near_wld['NEAR_FID'] == 113]


# number of cluster
ncluster = len(near_wld.NEAR_FID.unique())
ncluster
clusterid = near_wld.NEAR_FID.unique()

sed_max = []

cluster_df = near_wld[near_wld['NEAR_FID'] == 433]
nparcel = len(cluster_df.index)
topn = int(np.ceil(len(cluster_df.index)/2))
topn_cluster = cluster_df.nlargest(topn,'SedRed')
len(topn_cluster.index)

bot_cluster = cluster_df.nsmallest(nparcel-topn,'SedRed')
bot_cluster['SedRed'] = 0
bot_cluster['SedRed']

for j in clusterid:
    cluster_df = near_wld[near_wld['NEAR_FID'] == j]
    cluster_df.reset_index()
    indexmax = cluster_df['SedRed'].idxmax()
    sed_max.append(cluster_df.loc[indexmax]['SedRed'])
    

sed_max



### cluster creation
clusterid = near_wld.NEAR_FID.unique()
clusterid
ncluster = len(clusterid)
ncluster
def near_cluster (near_id):
    cluster = near_id
    sites = near_wld[near_wld.NEAR_FID == near_id].Site_ID.values
    sites_size = len(sites)
    distances = near_wld[near_wld.NEAR_FID == near_id].NEAR_DIST.values
    nearest_dist = np.min (distances)
    farest_dist = np.max (distances)
    sedcoeff = near_wld[near_wld.NEAR_FID == near_id].SedRed.values
    area = near_wld[near_wld.NEAR_FID == near_id].area_m2.values
    cluster_repr = sites[np.argmin (distances)]
    return (cluster,cluster_repr,nearest_dist,farest_dist,sites, sites_size, sedcoeff, area)
#near_wld[near_wld.NEAR_FID == 13751]
cluster_id = []
cluster_repre =[]
nearest_dis = []
farest_dis = []
sites = []
size = []
sed = []
areas = []
for i in clusterid:   
    cluster_id.append(near_cluster(i)[0])
    cluster_repre.append( near_cluster(i)[1])
    nearest_dis.append(near_cluster(i)[2])
    farest_dis.append(near_cluster(i)[3])
    size.append(near_cluster(i)[5])
    sites.append(near_cluster(i)[4])
    sed.append(near_cluster(i)[6])
    areas.append(near_cluster(i)[7])

near_cluster(7300)
cluster_dict = {'ClusterID':cluster_id,
                #'Cluster_RP': cluster_repre,
                #'ND': nearest_dis,
                #'FD':farest_dis,
                'Size':size,
                'Sites': sites,
                'SedRed': sed,
                'AreaM2': areas}
cluster_dict
cluster_df = pd.DataFrame(cluster_dict)
cluster_df.to_csv(data_folder/'output/prep_MAP_sub29.csv', index = False)

############cluster_df 

cluster_df.columns
cluster_df.index
cluster_df['zipred'] = 0
cluster_df['zipred']

for i in range(141):
    cluster_df['zipred'][i] = list(map(list, zip(cluster_df['Sites'][i],cluster_df['SedRed'][i])))
    
cluster_df['zipred'].head()
DF_list = list()

DF_list = [pd.DataFrame( cluster_df['zipred'][i]) for i in range(141)] 

DF_list
df_all = pd.concat(DF_list)
df_all
df_all.index


####################################
##if statement for epstasis 
SED = cluster_df.iloc[8]['SedRed']
SED
len(SED)
np.argsort(-SED)
SedRank= []
Sedeps = []
for i in range(ncluster):
    SED = cluster_df.iloc[i]['SedRed']
    SEDcopy =  SED.copy()
    ## If the selected number of parcels are greater than Topn, the rest of sed
    Topn = int(len(SED)/2)
    ## Get the index of the topn 
    ind = np.argpartition(SED, -Topn)[-Topn:]
    SEDcopy[~ind] = 0
    SedRank.append(ind)
    Sedeps.append(SEDcopy)

SED = cluster_df.iloc[5]['SedRed']  
SED 
SEDcopy =  SED.copy()
Topn = int(np.ceil(len(SED)/2))
Topn
ind = np.argpartition(SED, -Topn)[-Topn:]
ind

SEDcopy[-ind]
SEDcopy[~ind] = 0
[0 for x in SED if x not in ind]

cluster_df['SedRank'] = SedRank
cluster_df['SedEps'] = Sedeps
cluster_df.iloc[6]['SedRank']