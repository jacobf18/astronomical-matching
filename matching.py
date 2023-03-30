import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm
from sklearn.cluster import KMeans
import itertools
from gurobipy import Model, GRB, quicksum
from scipy.spatial.distance  import pdist
import itertools
from collections import defaultdict

from cop_kmeans import cop_kmeans
import constants


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

def run_cop_kmeans(data_df: pd.DataFrame, 
                   min_k: int = 1, 
                   max_k: int = 50, 
                   num_repeat = 1,
                   verbose = False) -> Tuple[List[int],int,float]:
    best_labels = []
    best_k = -1
    best_bayes = float('inf')
    loop = range(num_repeat)
    if verbose:
        loop = tqdm(range(num_repeat))
    for _ in loop:
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

def setup_miqcp_model(data_df, max_clusters = -1, min_clusters = 0, verbose = False):
    num_datapoints = data_df.shape[0]
    num_clusters = max_clusters
    if max_clusters == -1:
        num_clusters = num_datapoints
    num_catalogs = data_df["ImageID"].unique().shape[0]
    dims = 2

    C = np.log(constants.ARCSEC_TO_RAD_2) # constant used for arcseconds to radians conversion

    model = Model("MIQCP")
    if not verbose:
        model.setParam("OutputFlag", 0)

    # Make intermediate lists and dictionaries
    candidate_list = []
    coord_dict = dict()
    kappa_dict = dict()

    for _, row in data_df.iterrows():
        source_image = (row.SourceID, row.ImageID)
        candidate_list.append(source_image)
        coord_dict[source_image] = (row["coord1 (arcseconds)"],row["coord2 (arcseconds)"])
        kappa_dict[source_image] = row["kappa"]

    # Add cluster variables
    cluster_vars = model.addVars(num_clusters, dims, lb = -float('inf'), ub=float('inf'))

    # Add boolean variables (cluster-sources)
    x = model.addVars(candidate_list, 
                      list(range(num_clusters)), 
                      vtype=GRB.BINARY, 
                      name="x")

    for (source, catalog, k), var in x.items():
        var.setAttr("BranchPriority", 1)

    # cluster distances
    p = model.addVars(num_clusters, lb = 0, vtype=GRB.BINARY)

    # Add M variable
    M = np.max(pdist(data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]])) * data_df["kappa"].max() * 1.1
    # M = 10**6

    # Add max cluster distance variables
    r_dict = model.addVars(candidate_list, lb = 0.0, ub = float('inf'))

    # Log term
    error_threshold = 1/10 * np.log(data_df["kappa"].min())

    var_chi_dict = {}
    sigma_max = data_df["Sigma"].max()
    sigma_min = data_df["Sigma"].min()

    # b_list = [np.log(1/(sigma_max)**2) + C]
    # b_list = [np.log(1/(sigma_max)**2)]
    b_list = [(-2) * np.log(sigma_max)]
    # Compute b_list
    # while b_list[-1] < np.log(num_catalogs) - np.log((sigma_min)**2) + C:
    while b_list[-1] < np.log(num_catalogs) - (2*np.log((sigma_min))):
        b_list.append(b_list[-1]+error_threshold)
        
    num_breakpoints = len(b_list) # = P in the paper

    # Variables for chi
    for j in range(num_clusters):
        for b_i in range(num_breakpoints):
            var_chi_dict[('chi', j, b_i)] = model.addVar(vtype=GRB.BINARY, name=str(('chi', j, b_i)))

    s = model.addVars(num_clusters, lb = 0, vtype=GRB.INTEGER)

    ### Objective ###
    sum_ln_kappa_rad = (C * num_datapoints) + np.log(data_df["kappa"]).sum()
    model.setObjective((0.5 * r_dict.sum()) 
                        + (np.log(2) * p.sum()) 
                        - (np.log(2) * s.sum())
                        + quicksum((b_list[0] * var_chi_dict[('chi', j, 0)])
                                    + (error_threshold * quicksum(var_chi_dict[('chi', j, b_i)] 
                                                                    for b_i in range(1,num_breakpoints)))
                                    for j in range(num_clusters))
                        + (p.sum() * C)
                        - sum_ln_kappa_rad
                        , GRB.MINIMIZE)

    ### Constraints
    # Each point assigned to a cluster
    for source, catalog in candidate_list:
        model.addConstr(quicksum(x[(source, catalog, j)] for j in range(num_clusters)) == 1)

    # |# objects|
    # p = 1 if there is a source in that cluster
    # p = 0 if no sources assigned to cluster
    for j in range(num_clusters):
        for source, catalog in candidate_list:
            model.addConstr(p[j] >= x[source,catalog,j])

    # lower bound on objects for getting a maximum object count
    model.addConstr(p.sum() >= min_clusters)
            
    # |# sources| * ln(2)
    for j in range(num_clusters):
        model.addConstr(s[j] == quicksum(x[source,catalog,j] for source,catalog in candidate_list))

    # Each cluster has at most one source from a catalog
    sources_by_catalog = defaultdict(list)
    for source, catalog in candidate_list:
        sources_by_catalog[catalog].append(source)

    for j,c in itertools.product(range(num_clusters), range(num_catalogs)):
        model.addConstr(quicksum(x[(source,c,j)] for source in sources_by_catalog[c]) <= 1)

    # Min and max for cluster variables
    # Get coordinates
    x_coords = data_df["coord1 (arcseconds)"]
    y_coords = data_df["coord2 (arcseconds)"]

    for j in range(num_clusters):
        model.addConstr(cluster_vars[j,0] == [min(x_coords), max(x_coords)])
        model.addConstr(cluster_vars[j,1] == [min(y_coords), max(y_coords)])

    # Break symmetry
    first_s, first_c = candidate_list[0]
    model.addConstr(x[(first_s, first_c, 0)] == 1)

    # Big-M constraints
    for (source, catalog), coord in coord_dict.items():
        for j in range(num_clusters):
            model.addQConstr((kappa_dict[(source,catalog)] * # in arcseconds^-2
                                (((cluster_vars[j,0] - coord[0]) * (cluster_vars[j,0] - coord[0])) + 
                                ((cluster_vars[j,1] - coord[1]) * (cluster_vars[j,1] - coord[1])))) # in arcseconds ^ 2
                                <= 
                                r_dict[(source, catalog)] + (M * (1 - x[(source, catalog, j)])))

    # Definition of variables chi
    # Equation B19
    for j in range(num_clusters):
        chi_constraint_with_b = []
        chi_constraint = []
        x_constraint = []
        for breakpoint_index in range(1, num_breakpoints):
            chi_constraint_with_b.append(var_chi_dict[('chi', j, breakpoint_index)]*(np.exp(b_list[breakpoint_index])-np.exp(b_list[breakpoint_index-1])))
        for source, catalog in candidate_list:
            x_constraint.append(x[(source, catalog, j)]*kappa_dict[(source, catalog)]) 
        model.addConstr(np.exp(b_list[0])* var_chi_dict[('chi', j, 0)] + quicksum(variable for variable in chi_constraint_with_b) >= quicksum(variable for variable in x_constraint))
        
        for breakpoint_index in range(num_breakpoints):
            chi_constraint.append(var_chi_dict[('chi', j, breakpoint_index)])
        for chi_index in range(len(chi_constraint) - 1):
            model.addConstr(chi_constraint[chi_index] >= chi_constraint[chi_index + 1])

    model.setParam("NumericFocus", 2) # for numerical stability

    return model, x

def find_max_clusters(data_df) -> int:
    """Find the maximum number of clusters by bounding and relaxing the model and comparing
    the best possible bounded bayes factor to the feasible point returned by constrained
    kmeans.

    Args:
        data_df (pd.DataFrame): dataframe of coordinates and uncertainties

    Returns:
        int: maximum number of clusters
    """
    _, _, cop_kmeans_bayes = run_cop_kmeans(data_df, min_k=1,max_k = data_df.shape[0])
    max_cluster = 0
    for c in range(data_df.shape[0]):
        model,_ = setup_miqcp_model(data_df, max_clusters = -1, min_clusters=c)

        model.update()
        model.setParam("OutputFlag", 0)
        presolved = model.presolve()
        relaxed = presolved.relax()
        relaxed.setParam("OutputFlag", 0)

        relaxed.optimize()

        if relaxed.ObjVal > cop_kmeans_bayes:
            max_cluster = c - 1
            break
    return max_cluster

def miqcp(data_df: pd.DataFrame, 
          max_clusters = -1, 
          verbose = False, 
          preDual = True, 
          preQLinearize = True):
    
    max_clusters = find_max_clusters(data_df)

    model,x = setup_miqcp_model(data_df, max_clusters,0)

    ### Solve MIQCP
    if preDual:
        model.setParam("PreDual", 1)
    if preQLinearize:
        model.setParam("PreQLinearize", 2)

    model.optimize()

    labels = []
    for (_, _, j), var in x.items():
        if var.X > 0.5:
            labels.append(j)
    assert len(labels) == data_df.shape[0] # sanity check that every point is assigned a label
    
    return pd.factorize(labels)[0] # factorize sets labels from 0 to max_labels