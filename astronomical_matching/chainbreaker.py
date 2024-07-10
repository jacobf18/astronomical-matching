import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore

from .utils import neg_log_bayes, neg_log_bayes_adjusted


def chain_breaking(data_df: pd.DataFrame):
    """Single-linkage heuristic (chain-breaking)

    Args:
        data_df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    coords = data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]
    linkage_arr = linkage(coords, method="single")
    num_sources = data_df.shape[0]

    best_labels = None
    best_bayes = float("inf")
    best_k = -1
    for n_clusters in range(1, num_sources):
        cluster_labels = (
            fcluster(linkage_arr, t=n_clusters, criterion="maxclust") - 1
        )  # -1 because indexes from 1
        bayes = neg_log_bayes_adjusted(data_df, cluster_labels)
        if bayes < best_bayes:
            best_labels = cluster_labels
            best_bayes = bayes
            best_k = n_clusters
    return best_labels, best_k, best_bayes
