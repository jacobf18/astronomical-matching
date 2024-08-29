from typing import Union

import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from .utils import neg_log_bayes, neg_log_bayes_adjusted
from .miqcp import find_max_clusters


def run_kmeans(
    data_df: pd.DataFrame, min_k: int = None, max_k: int = None, verbose=False, corrected = True
) -> tuple[list[int], int, float]:
    coords = data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]
    weights = data_df["kappa"]

    best_labels = None
    best_k = 0
    best_bayes = float("inf")

    if max_k is None:
        max_k = find_max_clusters(data_df)
    if min_k is None:
        min_k = 1
    
    loop: Union[range, tqdm] = range(min_k, max_k)
    
    if verbose:
        loop = tqdm(range(min_k, max_k))
    for k in loop:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=100).fit(
            X=coords, sample_weight=weights
        )

        if corrected:
            bayes = neg_log_bayes_adjusted(data_df, kmeans.labels_)
        else:
            bayes = neg_log_bayes(data_df, kmeans.labels_)

        if bayes < best_bayes:
            best_bayes = bayes
            best_labels = kmeans.labels_
            best_k = k
    return best_labels, best_k, best_bayes
