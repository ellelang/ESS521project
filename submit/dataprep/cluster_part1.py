from __future__ import (absolute_import, division, print_function)
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
from shapely.geometry import Point, Polygon
from matplotlib.lines import Line2D
wcmodata = pd.read_csv(data_folder/'csv/WCMO.csv')
wcmodata.shape
near_wld = wcmodata[['Site_ID','NEAR_FID', 'NEAR_DIST']].sort_values(by=['NEAR_FID','NEAR_DIST'])
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
    cluster_repr = sites[np.argmin (distances)]
    return (cluster,cluster_repr,nearest_dist,farest_dist,sites,sites_size)
near_wld[near_wld.NEAR_FID == 13751]
cluster_id = []
cluster_repre =[]
nearest_dis = []
farest_dis = []
size = []
for i in clusterid:   
    cluster_id.append(near_cluster(i)[0])
    cluster_repre.append( near_cluster(i)[1])
    nearest_dis.append(near_cluster(i)[2])
    farest_dis.append(near_cluster(i)[3])
    size.append(near_cluster(i)[5])

cluster_id
cluster_repre
nearest_dis

cluster_dict = {'ClusterID':cluster_id,
                'Cluster_RP': cluster_repre,
                'ND': nearest_dis,
                'FD':farest_dis,
                'Size':size}
cluster_dict
cluster_df = pd.DataFrame(cluster_dict)
cluster_df
cluster_df.to_csv('nearID_represent.csv', index = False)

keys = ['cluster_id', 'Site_ID']
cluster_sites = {new_list: [] for new_list in keys} 
for i in clusterid:
    cluster_sites['cluster_id'].append( near_cluster(i)[0])
    cluster_sites['Site_ID'] .append(near_cluster(i)[4].tolist())
cluster_sites_df = pd.DataFrame(cluster_sites)

cluster_sites_df.to_csv('clusters_sites.csv', index = False)


