from typing import Union

import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from .utils import neg_log_bayes


def run_kmeans(
    data_df: pd.DataFrame, min_k: int, max_k: int = None, verbose=False
) -> tuple[list[int], int, float]:
    coords = data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]
    weights = data_df["kappa"]

    best_labels = None
    best_k = 0
    best_bayes = float("inf")

    if max_k is None:
        max_k = data_df.shape[0]
    
    loop: Union[range, tqdm] = range(min_k, max_k)
    
    if verbose:
        loop = tqdm(range(min_k, max_k))
    for k in loop:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=100).fit(
            X=coords, sample_weight=weights
        )

        bayes = neg_log_bayes(data_df, kmeans.labels_)

        if bayes < best_bayes:
            best_bayes = bayes
            best_labels = kmeans.labels_
            best_k = k
    return best_labels, best_k, best_bayes
