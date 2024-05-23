from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import umap
from sklearn.metrics import silhouette_score,davies_bouldin_score
import seaborn as sns
from matplotlib.colors import ListedColormap

Base= pd.read_excel('Complete_database.xlsx',index_col=0)
Base=Base.drop([12,18,23,24],axis=0)
Base1=Base.drop(['Pulf','EVA','FIQ'],axis=1)
Base=Base.drop(['Pulf','EVA','FIQ','Fibromialgia'],axis=1)
reducer = umap.UMAP(random_state=0, n_neighbors=4, min_dist=0.005, n_components=2)
Base=StandardScaler().fit_transform(Base)
embedding = reducer.fit_transform(Base)
figure, axis = plt.subplots(1, 1)
figure.set_size_inches(12, 7)
colors = ListedColormap(['red', 'blue'])

scatter=plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Base1.Fibromialgia, cmap=colors)
plt.xticks(fontname='serif',fontsize=17)
plt.yticks(fontname='serif',fontsize=17)
plt.legend(handles=scatter.legend_elements()[0], labels=['Control','Fibromyalgia'])
plt.rc('legend',fontsize=40)
# plt.title('UMAP projection ', fontsize=20);
plt.xlabel('UMAP_1',fontname='serif',fontsize=19)
plt.ylabel('UMAP_2',fontname='serif',fontsize=19)
plt.show()
plt.savefig('Figures/FM_Control_UMAP.png',bbox_inches='tight')
plt.close()


Base= pd.read_excel('Complete_database.xlsx',index_col=0)
Base= Base.drop(['EVA','FIQ','Pulf'],axis=1)
names= Base.drop(['Fibromialgia'],axis=1)
names= list(names.columns)
values_0_1=[]
for i in names:
    z_statistic, p_value = mannwhitneyu(Base[Base.Fibromialgia==0][i],Base[Base.Fibromialgia==1][i])
    print("The p value for the parameter",i,"is",p_value)
    if p_value < 0.05:
        print("We can reject the null hypothesis in the attribute ",i)
        values_0_1.append(i)
    else:
        print("We can not reject the null hypothesis in the attribute ",i)

for e in values_0_1:
    print(Base[[e,'cluster_label']].groupby('cluster_label').agg({'mean','count','std'}).round(3))