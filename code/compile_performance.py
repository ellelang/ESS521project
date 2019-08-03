import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data')

df_maple29_1000 = pd.read_csv(data_folder/'output/maple29_performance.csv')
df_maple29_3000 = pd.read_csv(
    data_folder/'output/maple29_performance_3000GEN.csv')
df_mapleWhole_500 = pd.read_csv(
    data_folder/'output/maple_performance_500GEN.csv')
df_mapleWhole_1000 = pd.read_csv(
    data_folder/'output/maple_performance_1000GEN.csv')


df_performance = pd.concat(
    [df_maple29_1000, df_maple29_3000, df_mapleWhole_500, df_mapleWhole_1000])
df_performance

df_performance.to_csv(
    data_folder/'output/df_performance_0802.csv', index=False)
