import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
from pathlib import Path
import argparse
import consts as consts
from consensus import perform_consensus_clustering
from plotter import plot_cdf_cc, plot_change_area_under_cdf, plot_auc_cdf, plot_hist_density_cc, plot_clustermap, \
    plot_umap_projections_with_cluster_label

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def find_number_clusters() -> list:
    list_sc = []
    list_dbi = []

    names = ['Silhouette score', 'Davies_Bouldin index']

    for num_clusters in [2, 3, 4, 5]:
        spectral_model = SpectralClustering(random_state=2, n_neighbors=5, affinity='nearest_neighbors',
                                            n_clusters=num_clusters)
        labels_rbf = spectral_model.fit_predict(embedding)
        list_sc.append(silhouette_score(embedding, labels_rbf))
        list_dbi.append(davies_bouldin_score(embedding, labels_rbf))

    cvi = [list_sc, list_dbi]

    return cvi


def normalize_data(df_data: pd.DataFrame, type_normalization: str) -> pd.DataFrame:
    if type_normalization == 'standard':
        df_data_normalized = StandardScaler().fit_transform(df_data)
    elif type_normalization == 'minmax':
        df_data_normalized = MinMaxScaler().fit_transform(df_data)
    else:
        type_normalization = StandardScaler().fit_transform(df_data)

    return df_data_normalized


def parse_arguments(parser):
    parser.add_argument('--clustname', default='ahc', type=str)
    parser.add_argument('--normalization', default='standard', type=str)
    parser.add_argument('--n_jobs', default=1, type=int)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='consensus clustering experiments')
args = parse_arguments(cmd_parser)

# Load data with features extracted
path_dataset = str(Path.joinpath(consts.PATH_PROJECT_DATA, 'Complete_database_{}.xlsx'.format('fibro')))
df_data = pd.read_excel(path_dataset, index_col=0)

# Remove patients with missing data in the acquisition data procedure
df_data = df_data.drop([12, 18, 23, 24], axis=0)

X = df_data[df_data['Fibromialgia'] == 1]
X = X.drop(['Fibromialgia', 'Pulf'], axis=1)

print(X.head())

# Normalize data
df_data_normalized = normalize_data(X, type_normalization=args.normalization)

# Perform feature reduction using the graph-based and local method UMAP
umap_reducer = umap.UMAP(n_neighbors=3, n_components=2, min_dist=0.01, random_state=1)
embedding = umap_reducer.fit_transform(df_data_normalized)
spectral_model = SpectralClustering(random_state=0, n_neighbors=5, affinity='nearest_neighbors', n_clusters=2)
labels_rbf = spectral_model.fit_predict(embedding)
plot_umap_projections_with_cluster_label(embedding, labels_rbf)

# Perform consensus clustering with different clustering methods
cc_obj = perform_consensus_clustering(embedding, clust_name=args.clustname, n_jobs=args.n_jobs)
best_n_clusters = cc_obj.best_k('knee')

plot_cdf_cc(cc_obj)
plot_change_area_under_cdf(cc_obj)
plot_auc_cdf(cc_obj)
plot_hist_density_cc(cc_obj, best_n_clusters)
plot_clustermap(cc_obj, n_clusters=best_n_clusters)



