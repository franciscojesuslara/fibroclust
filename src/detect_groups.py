import pandas as pd

from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import argparse

from sklearn.cluster import AgglomerativeClustering
from consensusclustering import ConsensusClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


from cclust import ConsensusCluster
from consensus import perform_consensus_clustering
from plotter import plot_cdf_cc, plot_change_area_under_cdf, plot_auc_cdf, plot_hist_density_cc, plot_clustermap, \
    plot_umap_projections_with_cluster_label

from utils import load_dataset, normalize_data

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def find_number_clusters() -> list:
    list_sc = []
    list_dbi = []

    names = ['Silhouette score', 'Davies_Bouldin index']

    for num_clusters in [2, 3, 4, 5]:
        spectral_model = SpectralClustering(random_state=2,
                                            n_neighbors=5,
                                            affinity='nearest_neighbors',
                                            n_clusters=num_clusters)
        labels_rbf = spectral_model.fit_predict(embedding)
        list_sc.append(silhouette_score(embedding, labels_rbf))
        list_dbi.append(davies_bouldin_score(embedding, labels_rbf))

    cvi = [list_sc, list_dbi]

    return cvi


def parse_arguments(parser):
    parser.add_argument('--clustname', default='ahc', type=str)
    parser.add_argument('--normalization', default='standard', type=str)
    parser.add_argument('--n_resamples', default=100, type=int)
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--save_figure', default=False, type=bool)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='consensus clustering experiments')
args = parse_arguments(cmd_parser)

# Load and normalize data
X = load_dataset()
df_data_normalized = normalize_data(X, type_normalization=args.normalization)

# Perform consensus clustering algorithm
cc = ConsensusCluster(cluster=SpectralClustering, L=2, K=8, H=args.n_resamples, resample_proportion=0.8)
cc.fit(df_data_normalized)

logger.info('The best k identified by consensus clustering: {}'.format(cc.bestK))

# Perform consensus clustering with different clustering methods
cc_obj = perform_consensus_clustering(df_data_normalized,
                                      clust_name=args.clustname,
                                      n_resamples=args.n_resamples,
                                      n_jobs=args.n_jobs
                                      )

best_n_clusters = cc_obj.best_k('knee')

# Obtain plots associated with consensus clustering
plot_cdf_cc(cc_obj, best_n_clusters, save_figure=args.save_figure)
plot_change_area_under_cdf(cc_obj, best_n_clusters, save_figure=args.save_figure)
plot_auc_cdf(cc_obj, best_n_clusters, save_figure=args.save_figure)
plot_hist_density_cc(cc_obj, best_n_clusters, save_figure=args.save_figure)
plot_clustermap(cc_obj, n_clusters=2, save_figure=args.save_figure)
plot_clustermap(cc_obj, n_clusters=3, save_figure=args.save_figure)
plot_clustermap(cc_obj, n_clusters=4, save_figure=args.save_figure)
plot_clustermap(cc_obj, n_clusters=5, save_figure=args.save_figure)

# Perform feature reduction using the graph-based and local method UMAP
umap_reducer = umap.UMAP(n_neighbors=3, n_components=2, min_dist=0.01, random_state=4242)
embedding = umap_reducer.fit_transform(df_data_normalized)
spectral_model = SpectralClustering(random_state=0, n_neighbors=5, affinity='nearest_neighbors', n_clusters=2)
labels_rbf = spectral_model.fit_predict(embedding)
plot_umap_projections_with_cluster_label(embedding, labels_rbf)


