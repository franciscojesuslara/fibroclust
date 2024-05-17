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


## Unsupervised learning (UMAP + sh+ db)

Base= pd.read_excel('Complete_database.xlsx',index_col=0)
Base=Base.drop([12,18,23,24],axis=0)
X=Base[Base['Fibromialgia']==1]
X= X.drop(['Fibromialgia','Pulf'],axis=1)
X=StandardScaler().fit_transform(X)
reducer = umap.UMAP(n_neighbors=3 ,n_components=2,min_dist=0.01,random_state=0)
embedding= reducer.fit_transform(X)
ss=[]
ch=[]
names=['Silhouette score','Davies_Bouldin index']
for e in [2,3,4,5]:
    spectral_model= SpectralClustering(random_state=2,n_neighbors=5, affinity='nearest_neighbors', n_clusters=e)
    labels_rbf = spectral_model.fit_predict(embedding)
    ss.append(silhouette_score(embedding, labels_rbf))
    ch.append(davies_bouldin_score(embedding, labels_rbf))
cvi=[ss,ch]
figure, axis = plt.subplots(1, 1)
figure.set_size_inches(12, 7)
plt.plot(np.arange(2,6,1), cvi[0])
plt.ylabel('Silhouette score',fontsize=23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Number of clusters',fontsize=23)
plt.savefig('Figures/sh.png',bbox_inches='tight')
plt.close()

figure, axis = plt.subplots(1, 1)
figure.set_size_inches(12, 7)
plt.plot(np.arange(2,6,1), cvi[1])
plt.ylabel('Davies_Bouldin index',fontsize=23)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Number of clusters',fontsize=23)
plt.savefig('Figures/db.png',bbox_inches='tight')
plt.close()

# ## Unsupervised learning (UMAP + spectral clustering)

colors=ListedColormap(['red','blue'])
clusterable_embedding = umap.UMAP(n_neighbors=3 ,n_components=2,min_dist=0.01,random_state=0).fit_transform(X)
spectral_model= SpectralClustering(random_state=0,n_neighbors=5, affinity='nearest_neighbors', n_clusters=2)
labels_rbf = spectral_model.fit_predict(clusterable_embedding)
fig = plt.figure(figsize=(10, 8))

scatter=plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
            c=labels_rbf, s=20, cmap=colors)
plt.xticks(fontname='serif', fontsize=16)
plt.yticks(fontname='serif', fontsize=16)
legend=plt.legend(handles=scatter.legend_elements()[0],labels=['FM Group 0','FM Group 1'],fontsize=20)

legend._legend_box.width = 250
plt.xlabel('UMAP_1', fontname='serif', fontsize=17)
plt.ylabel('UMAP_2', fontname='serif', fontsize=17)
plt.savefig('Figures/FM groups.png',bbox_inches='tight')
plt.close()
Fibro=Base[Base['Fibromialgia']==1]
Fibro['cluster_label']=labels_rbf
Fibro.to_csv('FM groups.csv',index=False)

#check differences between clusters

Fibro= pd.read_csv('FM groups.csv')
Fibro= Fibro.drop(['Fibromialgia','Pulf'],axis=1)
names= Fibro.drop(['cluster_label'],axis=1)
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

for e in values_0_1:
    print(Fibro[[e,'cluster_label']].groupby('cluster_label').agg({'mean','count','std'}).round(3))



