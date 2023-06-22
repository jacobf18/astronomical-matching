from typing import Union

import numpy as np
import pandas as pd

from .constants import ARCSEC_TO_RAD_2


def neg_log_bayes(data_df: pd.DataFrame, labels: Union[list[int], np.ndarray]) -> float:
    """Calculate negative log bayes factor using the given labels.

    Args:
        data_df (pd.DataFrame):
            Data dataframe with coordinates and uncertainties
        labels (list[int]): integer array of labels

    Returns:
        float: negative log bayes factor
    """
    out = 0
    # preprocess labels to make sure 0 through max number of labels
    labels = pd.factorize(labels)[0]

    # Set column of dataframe for indexing by label
    data_df["labels"] = labels

    for i in range(max(labels) + 1):
        # filter data to sources labeled to i
        data_labeled = data_df[data_df.labels == i]

        num_sources = data_labeled.shape[0]

        # get 2D coordinates of sources
        # coords = data_labeled[["coord1 (arcseconds)", "coord2 (arcseconds)"]]
        coord1 = data_labeled["coord1 (arcseconds)"]
        coord2 = data_labeled["coord2 (arcseconds)"]
        weights = data_labeled["kappa"]

        weights_sum = weights.sum()

        # Get centroid
        centroid1 = (coord1 * weights).sum() / weights_sum
        centroid2 = (coord2 * weights).sum() / weights_sum

        # Get number of sources
        num_sources = data_labeled.shape[0]

        # Get kappa sums
        # sum_ln_kappa_rad = data_labeled["log kappa (radians)"].sum()
        # ln_sum_kappa_rad = np.log(data_labeled["kappa (radians)"].sum())

        C = np.log(ARCSEC_TO_RAD_2)
        ln_sum_kappa_rad = C + np.log(weights_sum)
        sum_ln_kappa_rad = (C * num_sources) + np.log(weights).sum()

        # Get sum of pairwise distances between all sources
        square_dist_weighted = weights * (
            np.power(coord1 - centroid1, 2) + np.power(coord2 - centroid2, 2)
        )
        sum_of_square_dist = square_dist_weighted.sum()

        # Divide by 2
        double_sum = sum_of_square_dist / 2

        # Add up all terms
        out += (
            (1 - num_sources) * np.log(2)
            - sum_ln_kappa_rad
            + double_sum
            + ln_sum_kappa_rad
        )

    # Remove the label column
    del data_df["labels"]

    return out
