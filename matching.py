import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm
from sklearn.cluster import KMeans
from cop_kmeans import cop_kmeans
import constants
import itertools


def run_kmeans(data_df: pd.DataFrame, min_k: int, max_k: int) -> Tuple[List[int],int,float]:
    coords = data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]
    weights = data_df["kappa"]

    best_labels = None
    best_k = 0
    best_bayes = np.Inf

    for k in tqdm(range(min_k,max_k)):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init = 100).fit(X = coords, sample_weight=weights)

        bayes = neg_log_bayes(data_df, kmeans.labels_)

        if bayes < best_bayes:
            best_bayes = bayes
            best_labels = kmeans.labels_
            best_k = k
    return best_labels, best_k, best_bayes



def run_cop_kmeans_single(data_df: pd.DataFrame, min_k: int = 1, max_k: int = 50) -> Tuple[List[int],int,float]:
    """

    Args:
        data_df (pd.DataFrame): dataframe with coordinates, kappas, etc.

    Returns:
        tuple: labels, k, and negative log bayes factor
    """
    coords = data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]].to_numpy()
    weights = data_df["kappa"]

    # Create cannot-links (list of tuples)
    cannot_link_dict: Dict[int,List[int]] = dict()

    for i in range(max(data_df.ImageID)):
        cannot_link_dict[i] = data_df[data_df.ImageID == i].index.to_list()

    cannot_link = []

    for l in cannot_link_dict.values():
        for comb in itertools.combinations(l,2):
            cannot_link.append(comb)

    best_labels = None
    best_k = 0
    best_bayes = np.Inf

    for k in range(min_k,max_k):
        clusters, _ = cop_kmeans(dataset=coords, initialization="kmpp", k=k, ml=[],cl=cannot_link, sample_weights=weights)
        
        if clusters is None:
            continue

        bayes = neg_log_bayes(data_df, clusters)

        if bayes < best_bayes:
            best_bayes = bayes
            best_labels = clusters
            best_k = k
    
    return best_labels, best_k, best_bayes

def run_cop_kmeans(data_df: pd.DataFrame, min_k: int = 1, max_k: int = 50, num_repeat = 1) -> Tuple[List[int],int,float]:
    best_labels = []
    best_k = -1
    best_bayes = float('inf')
    for _ in tqdm(range(num_repeat)):
        labels, k, bayes = run_cop_kmeans_single(data_df, min_k, max_k)
        if bayes < best_bayes:
            best_labels = labels
            best_k = k
            best_bayes = bayes
    return best_labels, best_k, best_bayes


def tangent(ra: float, dec: float) -> Tuple[np.ndarray, np.ndarray]:
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
    """ Load the data into a pandas dataframe from a query that looks like this:

    select m.MatchID, m.Level, m.SubID, s.ImageID, l.SourceID, l.X, l.Y, l.Z, s.Sigma, s.RA, s.Dec
    from HSCv3.xrun.Matches m
            join HSCv3.xrun.MatchLinks l on l.MatchID=m.MatchID and l.Level=m.Level and l.JobID=m.JobID and l.SubID=m.SubID
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

    data_df["coord1 (arcseconds)"] = (data_df[["X", "Y", "Z"]] @ center_west) * 180 * 3600 / np.pi
    data_df["coord2 (arcseconds)"] = (data_df[["X", "Y", "Z"]] @ center_north) * 180 * 3600 / np.pi
    
    return data_df

def neg_log_bayes(data_df: pd.DataFrame, labels: List[int]) -> float:
    """ Calculate negative log bayes factor using the given labels.

    Args:
        data_df (pd.DataFrame): Data dataframe with coordinates and uncertainties
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
        
        C = np.log(constants.ARCSEC_TO_RAD_2)
        ln_sum_kappa_rad = C + np.log(weights_sum)
        sum_ln_kappa_rad = (C * num_sources) + np.log(weights).sum()

        # Get sum of pairwise distances between all sources
        square_dist_weighted = weights * (np.power(coord1 - centroid1, 2) + np.power(coord2 - centroid2, 2))
        sum_of_square_dist = square_dist_weighted.sum()
        
        # Divide by 2
        double_sum = sum_of_square_dist / 2
        
        # Add up all terms
        out += (1 - num_sources) * np.log(2) - sum_ln_kappa_rad + double_sum + ln_sum_kappa_rad
        
    # Remove the label column
    del data_df["labels"]

    return out