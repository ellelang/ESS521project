from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
SBIO = pd.read_csv(data_folder/'csv/SB IO Detail.csv')
SBIO.columns
wcmoga = pd.read_csv(data_folder/'csv/WCMOga.csv')
wcmoga.columns

wcmodata = pd.read_csv(data_folder/'csv/WCMO.csv')
wcmodata.columns
wcmosub = pd.merge(left = wcmodata, right = SBIO, left_on = 'HYDSB_LES30SB', right_on = 'SUBID', how = 'left')
wcmosub.shape
wcmoga_merge = pd.merge(left = wcmoga, right = wcmosub, left_on = 'ID', right_on = 'Site_ID')
wcmoga_merge.columns
#wcmoga_merge = wcmoga_merge.drop(['Type','HYDSB', 'SEDSB', 'SEDSB_R', ], axis=1)
wcmoga_merge.to_csv('wcmogawhole.csv', index = False)

icmoga = pd.read_csv(data_folder/'csv/ICMOga.csv')
icmoga.columns
icmodata = pd.read_csv(data_folder/'csv/ICMO.csv')
icmodata.columns

icmoga_merge = pd.merge(left = icmoga, right = icmodata, left_on = 'ID', right_on = 'ID')
icmoga_merge.columns
icmoga_merge.to_csv(data_folder/'icmogawhole.csv', index = False)

def River_sub (data, river_col, river_id):
    wcmo_riversub = data[data[river_col] == river_id]

    globals()['wcmo_%s' % river_id] =  wcmo_riversub 
    globals()['n%s' % river_id] =  wcmo_riversub.shape[0] 
    
    filename = 'wcmo_' + river_id + '.csv'
    wcmo_riversub.to_csv(filename, index = False)
 
riverid = ['LES','COB','MAP']

subid = list(range(1,31))
subid


for r in riverid:
    River_sub(wcmoga_merge, 'River_x', r)

wcmo_MAP.columns   
nLES
nMAP
nCOB





def Hyd_sub (data, sub_col, sid, river_id):
    wcmo_hydsub = data[data[sub_col] == sid] 
    globals()['n%s%d' % (river_id,sid)] =  wcmo_hydsub.shape[0]
    if wcmo_hydsub.shape[0] > 0:
        globals()['wcmo_%s_sub%d' % (river_id,sid)] = wcmo_hydsub
        #filename = 'wcmo_' + river_id + '_sub' + str(sid) + '.csv'
        #wcmo_hydsub.to_csv(filename, index = False)
    
for sid in subid:
    Hyd_sub (wcmo_LES, 'HYDSB_LES30SB', sid, 'LES')
    Hyd_sub (wcmo_COB, 'HYDSB_LES30SB', sid, 'COB')
    Hyd_sub (wcmo_MAP, 'HYDSB_LES30SB', sid, 'MAP')
    

NCOB = []
NMAP = []
NLES = []
for i in subid:
    NCOB.append(eval('nCOB%d'% (i)))
    NMAP.append(eval('nMAP%d'% (i)))
    NLES.append(eval('nLES%d'% (i)))

print(np.sum(NCOB))
print(np.sum(NMAP))
print(np.sum(NLES))
    






datafiles = [wcmo_LES, wcmo_COB, wcmo_MAP]
for data in datafiles:
    print(str(data))
##MAP RIVER  
for data in datafiles:
    for sid in subid:
    wcmo_hydsub = data[data['HYDSB_LES30SB'] == sid]
    globals()['wcmo_MAP_sub%d' % sid] = wcmo_hydsub
    globals()['nMAPsub%d' % sid] =  wcmo_hydsub.shape[0]
    if wcmo_hydsub.shape[0] > 0:
        filename = 'hysub' % data + str(sid) + '.csv'
        wcmo_hydsub.to_csv(filename, index = False)
    

        
    
##COB RIVER   
for sid in subid:
    wcmo_hydsub = wcmo_COB[wcmo_COB['HYDSB_LES30SB'] == sid]
    globals()['wcmo_COB_sub%d' % sid] = wcmo_hydsub
    globals()['nCOBsub%d' % sid] =  wcmo_hydsub.shape[0] 
    
##LES RIVER   
for sid in subid:
    wcmo_hydsub = wcmo_LES[wcmo_LES['HYDSB_LES30SB'] == sid]
    globals()['wcmo_LES_sub%d' % sid] = wcmo_hydsub
    globals()['nLESsub%d' % sid] =  wcmo_hydsub.shape[0] 
    

sub_id = list(range(1,31))
sub_id



