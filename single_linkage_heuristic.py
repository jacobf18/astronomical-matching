from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def neg_log_bayes_factor(cluster: np.array):
    """
    Calculate the negative log Bayes Factor for a given clustering.

    cluster is an numpy array with shape (n, 3) where n is the number of
    sources in the cluster.

    The first column is the first coordinate, the second column is the second, and
    the third column is the uncertainty.

    See Nguyen et al. 2022 for details on the Bayes factor.
    """
    num_objects = cluster.shape[0]

    kappas = np.reciprocal(cluster.T[2] ** 2)  # kappa = 1/sigma^2
    sum_kappas = np.sum(kappas)  # sum of all kappas

    # First 3 terms in formula
    neg_bayes = (
        (1 - num_objects) * np.log(2) - np.sum(np.log(kappas)) + np.log(sum_kappas)
    )

    # Double sum in formula
    pairwise_dist = pdist(
        cluster[:, :2], metric="sqeuclidean"
    )  # get pairwise squared distances
    kappas_prod = kappas[:, None] * kappas  # get pairwise products (vectorized)
    np.fill_diagonal(kappas_prod, 0)  # fill the diagonal with 0's for the next line
    kappas_prod_square = squareform(kappas_prod)  # convert to condensed distance vector

    # Add final term
    neg_bayes += np.sum(kappas_prod_square * pairwise_dist) / (4 * sum_kappas)

    return neg_bayes


def assign_cluster_neg_bayes(cluster_labels: np.array, data: np.array) -> float:
    """Calculate the total negative log bayes factor for a given clustering.

    Args:
        cluster_labels (np.array): n by 1 array of cluster labels.
        data (np.array): n by 3 array of data points.

    Returns:
        float: sum of negative log bayes factors for each cluster.
    """
    neg_bayes = 0
    for label in np.unique(cluster_labels):
        cluster = data[cluster_labels == label, :]
        neg_bayes += neg_log_bayes_factor(cluster)
    return neg_bayes


def find_clusters_single_linkage(
    cmat: np.array,
) -> tuple[np.array, int, float, list[float]]:
    """Find clusters using single linkage clustering."""

    data = np.concatenate(cmat, axis=0)
    linkage_arr = linkage(data[:, :2], method="single")

    bayes = []
    best_bayes = np.inf
    best_labels: np.array = None
    best_n: int = 0
    for n in range(75, data.shape[0]):
        cluster_labels = fcluster(linkage_arr, t=n, criterion="maxclust")
        bayes_factor = assign_cluster_neg_bayes(cluster_labels, data)
        bayes.append(bayes_factor)
        if bayes_factor < best_bayes:
            best_bayes = bayes_factor
            best_labels = cluster_labels
            best_n = n

    return best_labels, best_n, best_bayes, bayes
