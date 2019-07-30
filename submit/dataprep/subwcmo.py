from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
wcmodata_mapsub29 = pd.read_csv(data_folder/'output/wcmo_MAP_sub29.csv')
wcmodata_mapsub29.shape
near_wld = wcmodata_mapsub29[['Site_ID','SedRed','Cost','NEAR_FID', 'NEAR_DIST','area_m2']].sort_values(by=['NEAR_FID','SedRed'])
near_wld
near_wld.to_csv(data_folder/'output/nearcluster_MAP_sub29.csv', index = False)
near_wld
near_wld.columns
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
####################################
SED = cluster_df.iloc[8]['SedRed']
SED
len(SED)
np.argsort(-SED)
SedRank= []
Sedeps = []
for i in range(ncluster):
    SED = cluster_df.iloc[i]['SedRed']
    SEDcopy =  SED.copy()
    Topn = int(len(SED)/2)
    ind = np.argpartition(SED, -Topn)[-Topn:]
    SEDcopy[~ind] = 0
    SedRank.append(ind)
    Sedeps.append(SEDcopy)
cluster_df['SedRank'] = SedRank
cluster_df['SedEps'] = Sedeps
cluster_df.iloc[6]['SedRank']
#cluster_df['SedRank']
cluster_df.to_csv(data_folder/'output/nearID_wcmo_map29.csv', index = False)
