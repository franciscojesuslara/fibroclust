from sklearn.cluster import AgglomerativeClustering
from consensusclustering import ConsensusClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import numpy as np

import seaborn as sns
sns.set(style='ticks', font_scale=1.2)


def select_clustering_method(clust_name: str):
    if clust_name == 'ahc':
        clustering_obj = AgglomerativeClustering(metric='euclidean', linkage='ward')
    elif clust_name == 'spectral':
        clustering_obj = SpectralClustering(n_neighbors=4, affinity='nearest_neighbors')
    else:
        clustering_obj = KMeans(n_init="auto")

    return clustering_obj


def perform_consensus_clustering(m_data: np.array,
                                 clust_name: str = 'ahc',
                                 min_clusters: int = 2,
                                 max_clusters: int = 8,
                                 n_resamples: int = 500,
                                 n_jobs: int = 1):

    clustering_obj = select_clustering_method(clust_name)

    cc = ConsensusClustering(
        clustering_obj=clustering_obj,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        n_resamples=n_resamples,
        resample_frac=0.9,
        k_param='n_clusters'
    )

    cc.fit(m_data, progress_bar=True, n_jobs=n_jobs)

    return cc


