import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path
from consensusclustering import ConsensusClustering

import consts as consts

# figure, axis = plt.subplots(1, 1)
# figure.set_size_inches(12, 7)
# plt.plot(np.arange(2, 6, 1), cvi[0])
# plt.ylabel('Silhouette score', fontsize=23)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Number of clusters', fontsize=23)
# plt.savefig('Figures/sh.png', bbox_inches='tight')
# plt.close()
#
# figure, axis = plt.subplots(1, 1)
# figure.set_size_inches(12, 7)
# plt.plot(np.arange(2,6,1), cvi[1])
# plt.ylabel('Davies_Bouldin index',fontsize=23)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Number of clusters', fontsize=23)
# plt.savefig('Figures/db.png', bbox_inches='tight')
# plt.close()
#


def plot_cdf_cc(cc_obj: ConsensusClustering):
    _, ax = plt.subplots(figsize=(3.5, 3.5))
    cc_obj.plot_cdf(ax=ax)
    ax.legend(bbox_to_anchor=(1, 1))
    plt.show()


def plot_change_area_under_cdf(cc_obj: ConsensusClustering):
    _, ax = plt.subplots(figsize=(5, 3.5))
    cc_obj.plot_change_area_under_cdf(ax=ax)
    plt.show()

    cc_obj.best_k('change_in_auc')


def plot_auc_cdf(cc_obj: ConsensusClustering):
    cc_obj.plot_auc_cdf()
    plt.show()


def plot_hist_density_cc(cc_obj: ConsensusClustering, best_n_clusters: int):

    _, axes = plt.subplots(1, best_n_clusters, figsize=(12, 3.5))
    for i, ax in enumerate(axes):
        cc_obj.plot_hist(i + 2, ax=ax)
        ax.set_title(i + 2)

    plt.show()


def plot_clustermap(cc_obj: ConsensusClustering, n_clusters: int, figsize=(5, 5)):

    grid = cc_obj.plot_clustermap(
        k=n_clusters,
        figsize=figsize,
        dendrogram_ratio=0.05,
        xticklabels=False,
        yticklabels=False
    )
    grid.cax.set_visible(False)
    plt.show()


def plot_umap_projections_with_cluster_label(clusterable_embedding: np.matrix, labels_rbf: np.matrix):
    colors = ListedColormap(['red', 'blue'])
    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=labels_rbf, s=20, cmap=colors)
    plt.xticks(fontname='serif', fontsize=16)
    plt.yticks(fontname='serif', fontsize=16)
    # legend = plt.legend(handles=scatter.legend_elements()[0], labels=['FM Group 0', 'FM Group 1'], fontsize=20)
    # legend._legend_box.width = 250
    plt.xlabel('UMAP_1', fontname='serif', fontsize=17)
    plt.ylabel('UMAP_2', fontname='serif', fontsize=17)
    path_umap_fig = str(Path.joinpath(consts.PATH_PROJECT_FIGURES, 'umap_fm_groups.png'))
    plt.savefig(path_umap_fig, bbox_inches='tight')
    plt.close()



