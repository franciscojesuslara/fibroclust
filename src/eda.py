from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import umap
from sklearn.metrics import silhouette_score,davies_bouldin_score
import seaborn as sns
from matplotlib.colors import ListedColormap

import logging
import coloredlogs

import consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

Fibro = df_data[df_data['Fibromialgia'] == 1]
Fibro['cluster_label'] = labels_rbf
Fibro.to_csv('FM groups.csv',index=False)

#check differences between clusters
Fibro= pd.read_csv('../data/FM groups.csv')
Fibro= Fibro.drop(['Fibromialgia', 'Pulf'], axis=1)
names= Fibro.drop(['cluster_label'], axis=1)
names= list(names.columns)
values_0_1=[]

for i in names:
    z_statistic, p_value = mannwhitneyu(Fibro[Fibro.cluster_label==0][i],Fibro[Fibro.cluster_label==1][i])
    print("The p value for the parameter",i,"is",p_value)
    if p_value < 0.05:
        print("We can reject the null hypothesis in the attribute ",i)
        values_0_1.append(i)
    else:
        print("We can not reject the null hypothesis in the attribute ",i)

for num_clusters in values_0_1:
    print(Fibro[[num_clusters, 'cluster_label']].groupby('cluster_label').agg({'mean', 'count', 'std'}).round(3))



