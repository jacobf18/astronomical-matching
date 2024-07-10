import math
from functools import lru_cache
from typing import Union

import numpy as np
import pandas as pd

from .constants import ARCSEC_TO_RAD_2


@lru_cache
def stirling2(n, k):
    """
    Stirling number of the second kind.
    See https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind.
    """
    if n == k == 0:
        return 1
    if (n > 0 and k == 0) or (n == 0 and k > 0):
        return 0
    if n == k:
        return 1
    if k > n:
        return 0
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


def neg_log_bayes_adjusted(
    data_df: pd.DataFrame, labels: Union[list[int], np.ndarray]
) -> float:
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
            + math.log(stirling2(int(data_df.shape[0]), int(max(labels) + 1)))
        )

    # Remove the label column
    del data_df["labels"]

    return out

def neg_log_bayes(
    data_df: pd.DataFrame, labels: Union[list[int], np.ndarray]
) -> float:
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


def tangent(ra: float, dec: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns the tangent vectors pointing to west and north.

    Args:
        ra (float): right ascension in degrees
        dec (float): declination in degrees

    Returns:
        tuple[float,float]: west and north vectors
    """
    # Convert to radians
    ra = np.radians(ra)
    dec = np.radians(dec)

    # Get sin and cos of ra and dec
    sinRa = np.sin(ra)
    cosRa = np.cos(ra)
    sinDec = np.sin(dec)
    cosDec = np.cos(dec)

    # Get tangent vectors
    west = np.array([sinRa, -cosRa, 0])
    north = np.array([-sinDec * cosRa, -sinDec * sinRa, cosDec])
    return west, north


def load_data(file_path: str) -> pd.DataFrame:
    """ Load the data into a pandas dataframe
    from a query that looks like this:

    select (m.MatchID, m.Level, m.SubID, s.ImageID,
            l.SourceID, l.X, l.Y, l.Z, s.Sigma, s.RA, s.Dec)
    from HSCv3.xrun.Matches m
            join (HSCv3.xrun.MatchLinks l on l.MatchID=m.MatchID and
                  l.Level=m.Level and
                  l.JobID=m.JobID and
                  l.SubID=m.SubID)
            join HSCv3.whl.Sources s on s.SourceID=l.SourceID
    where m.JobID=______ and m.MatchID=__________ -- user input
            and m.Level=m.BestLevel -- for best results
    order by m.MatchID, m.SubID, s.ImageID, s.SourceID

    Args:
        file_path (str): path to match csv file

    Returns:
        pd.DataFrame: dataframe with columns from SQL query
    """
    data_df = pd.read_csv(file_path)

    # Convert Image IDs to integers in range [0, number of images]
    data_df.ImageID = pd.factorize(data_df.ImageID)[0]
    data_df.SourceID = pd.factorize(data_df.SourceID)[0]

    # Get kappas (inverse of sigma)
    data_df["Sigma"] = data_df["Sigma"] / np.sqrt(2.3)
    data_df["kappa"] = 1 / (data_df["Sigma"] ** 2)
    data_df["kappa (radians)"] = 1 / ((data_df["Sigma"]*np.pi/180/3600) ** 2)
    data_df["log kappa (radians)"] = np.log(data_df["kappa (radians)"])

    # Get center of data points
    center_ra = data_df.RA.mean()
    center_dec = data_df.Dec.mean()

    center_west, center_north = tangent(center_ra, center_dec)

    data_df["coord1 (arcseconds)"] = ((data_df[["X", "Y", "Z"]] @ center_west)
                                      * 180 * 3600 / np.pi)
    data_df["coord2 (arcseconds)"] = ((data_df[["X", "Y", "Z"]] @ center_north)
                                      * 180 * 3600 / np.pi)

    return data_df
