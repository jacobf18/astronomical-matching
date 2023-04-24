import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm
from sklearn.cluster import KMeans
import itertools
from gurobipy import Model, GRB, quicksum
from scipy.spatial.distance  import pdist
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

from cop_kmeans import cop_kmeans
import constants

def chain_breaking(data_df: pd.DataFrame, min_clusters = 1):
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
    best_bayes = np.inf
    best_k = -1
    for n in range(1, num_sources):
        cluster_labels = fcluster(linkage_arr, t = n, criterion = 'maxclust') - 1 # -1 because indexes from 1
        bayes = neg_log_bayes(data_df, cluster_labels)
        if bayes < best_bayes:
            best_labels = cluster_labels
            best_bayes = bayes
            best_k = n
    return best_labels, best_k, best_bayes

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

def plot_lined(labels, data_df):
    loc_tups = [(row["coord1 (arcseconds)"], row["coord2 (arcseconds)"]) for _, row in data_df.iterrows()]

    label_dict = defaultdict(list)
    for tup, label in zip(loc_tups, labels):
        label_dict[label].append(tup)
        
    plt.figure(figsize=(5,5))
    for label, tup_list in label_dict.items():
        if len(tup_list) == 1:
            pairs = [tup_list]
        else:
            pairs = list(itertools.combinations(tup_list, 2))
        for pair in pairs:
            x = []
            y = []
            for tup in pair:
                x.append(tup[0])
                y.append(tup[1])
            plt.plot(x, y, color='green', marker='o', linestyle='-', linewidth=1, markersize=2)

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
                   max_k: int = 100,
                   num_repeat = 1,
                   verbose = False) -> Tuple[List[int],int,float]:
    best_labels = []
    best_k = -1
    best_bayes = float('inf')
    loop = range(num_repeat)
    max_k = data_df.shape[0]
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
    model.setParam("NodefileStart", 2.0)

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
        model = model.presolve()
        model = model.relax()
        model.setParam("OutputFlag", 0)

        model.optimize()
        max_cluster = c - 1
        if model.ObjVal > cop_kmeans_bayes:
            break
    return max_cluster

def miqcp(data_df: pd.DataFrame,
          verbose = False, 
          preDual = False, 
          preQLinearize = False):
    
    max_clusters = find_max_clusters(data_df)
    # max_clusters = 2
    if verbose:
        print(f"Max Clusters using COP-KMeans: {max_clusters}")

    model, x = setup_miqcp_model(data_df, max_clusters,0)

    ### Solve MIQCP
    if preDual:
        model.setParam("PreDual", 1)
    if preQLinearize:
        model.setParam("PreQLinearize", 2)

    if verbose:
        model.setParam("OutputFlag", 1)
    model.optimize()

    labels = []
    for (_, _, j), var in x.items():
        if var.X > 0.5:
            labels.append(j)
    assert len(labels) == data_df.shape[0] # sanity check that every point is assigned a label
    
    return pd.factorize(labels)[0] # factorize sets labels from 0 to max_labels


def setup_dirilp(data_df: pd.DataFrame, max_clusters=-1, min_clusters = 0, verbose = False):
    """Setup DirILP model.

    Args:
        data_df (pd.DataFrame): dataframe with coordinates and uncertainties
        max_clusters (int, optional): maximum number of clusters. Defaults to -1.
        min_clusters (int, optional): minimum number of clusters. Defaults to 0.
        verbose (bool, optional): print out option. Defaults to False.

    Returns:
        tuple[model, vars]: model and binary decision variables
    """
    num_datapoints = data_df.shape[0]
    num_clusters = max_clusters
    if max_clusters == -1:
        num_clusters = num_datapoints
    num_catalog = data_df["ImageID"].unique().shape[0]
    # dims = 2

    # Make intermediate lists and dictionaries
    candidate_list = []
    coord_dict = dict()
    kappa_dict = dict()

    for _, row in data_df.iterrows():
        source_image = (row.SourceID, row.ImageID)
        candidate_list.append(source_image)
        coord_dict[source_image] = (row["coord1 (arcseconds)"],row["coord2 (arcseconds)"])
        kappa_dict[source_image] = row["kappa"]

    C = np.log(constants.ARCSEC_TO_RAD_2) # constant used for arcseconds to radians conversion

    sigma_min = data_df["Sigma"].min()
    sigma_max = data_df["Sigma"].max()

    mo = Model("likelihood")
    if not verbose:
        mo.setParam("OutputFlag", 0)

    M = np.ceil(1/sigma_min**4*sum(pdist(data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]))/(4*1/sigma_max**2))
    M_2 = num_catalog * max(kappa_dict.values())
    rounding_index = -1 #round number to nearest 10 if rounding_index = -1. to round to nearest 100, change to -2
    error_threshold = 1/ 100 * min(kappa_dict.values())

    t_list = []
    p_list = []
    b_list = [-2 * np.log(sigma_max)]
    c_list = [0, round(min(kappa_dict.values()),rounding_index)]
    # n = len(candidate_list)

    ########################### SET VARIABLES ###########################
    # tup_to_ind = {v: i for i, v in enumerate(candidate_list)}

    # Compute b_list
    while b_list[-1] < np.log(num_catalog) - (2*np.log(sigma_min)):
        b_list.append(b_list[-1]+error_threshold)

    num_breakpoints = len(b_list) # = P in the paper

    # Compute c_list
    max_c = round(num_catalog*max(kappa_dict.values()),rounding_index)
    while c_list[-1] < max_c:
        c_list.append(c_list[-1]+10**(-rounding_index))
        # c_list.append(c_list[-1]+(max_c/10**(-rounding_index)))
        
    # Variables for x
    x_dict = mo.addVars(num_clusters, candidate_list, vtype=GRB.BINARY)

    var_y_dict = {}
    # Variables for y
    for subset_index in range(num_clusters):
        for p1, p2 in list(itertools.combinations(candidate_list, r = 2)):
            if p1[1] != p2[1]:
                var_y_dict[('y', subset_index, p1, p2)] = mo.addVar(vtype=GRB.BINARY, name=str(('y', subset_index, p1,p2)))

    # Variables for z
    z_dict = mo.addVars(num_clusters, num_catalog + 1, vtype = GRB.BINARY)

    # Variables for t
    for subset_index in range(num_clusters):
        t_list.append(mo.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=str(('t', subset_index))))

    # Variables for u
    u_dict = mo.addVars(num_clusters, len(c_list), vtype=GRB.BINARY)
        
    # Variables for p
    for subset_index in range(num_clusters):
        p_list.append(mo.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(('p', subset_index))))

    # Variables for chi
    chi_dict = mo.addVars(num_clusters, num_breakpoints, vtype=GRB.BINARY)

    # Variables to count number of sources in a cluster
    p = mo.addVars(num_clusters, lb = 0, vtype=GRB.BINARY)

    ########################### SET OBJECTIVES ###########################

    # Set objective
    mo.setObjective(quicksum(p_list[subset_index]
                            + b_list[0]*chi_dict[subset_index, 0] + error_threshold*quicksum(chi_dict[subset_index, breakpoint_index] for breakpoint_index in range(1,num_breakpoints))
                            + t_list[subset_index] 
                            for subset_index in range(num_clusters))
                    + (p.sum() * C),
                    GRB.MINIMIZE)

    ########################### SET CONSTRAINTS ###########################

    # All detections (i,c) needs to belong to some subset (S_j)
    # Equation B14
    # for catalog_source_pair in candidate_list:
    #     x_constraint = []
    #     for j,s,c in x_dict.keys():
    #         if (s,c) == catalog_source_pair:
    #             x_constraint.append(x_dict[j,s,c])
    #     mo.addConstr(quicksum(variable for variable in x_constraint) == 1)
    #     # mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.EQUAL, rhs = 1)
    # Each point assigned to a cluster
    for source, catalog in candidate_list:
        mo.addConstr(quicksum(x_dict[j, source, catalog] for j in range(num_clusters)) == 1)

    # |# objects|
    # p = 1 if there is a source in that cluster
    # p = 0 if no sources assigned to cluster
    for j in range(num_clusters):
        for s,c in candidate_list:
            mo.addConstr(p[j] >= x_dict[j,s,c])

    mo.addConstr(p.sum() >= min_clusters)

    # Every subset takes no more than 1 detection from each catalog
    # # Equation B15
    # for subset_index in range(num_clusters):
    #     for catalog_index in range(num_catalog):
    #         x_constraint = []
    #         for j,s,c in x_dict.keys():
    #             if j == subset_index and c == catalog_index:
    #                 x_constraint.append(x_dict[j,s,c])
    #         mo.addConstr(quicksum(variable for variable in x_constraint) <= 1)
    #         # mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.LESS_EQUAL, rhs = 1)

    # Each cluster has at most one source from a catalog
    sources_by_catalog = defaultdict(list)
    for source, catalog in candidate_list:
        sources_by_catalog[catalog].append(source)

    for j,c in itertools.product(range(num_clusters), range(num_catalog)):
        mo.addConstr(quicksum(x_dict[j,source,c] for source in sources_by_catalog[c]) <= 1)

    # Definition of variables y
    # Equation B16
    for subset_index in range(num_clusters):
        for tup1, tup2 in list(itertools.combinations(candidate_list, r = 2)):
            if tup1[1] != tup2[1]:
                mo.addConstr(var_y_dict[('y', subset_index, tup1, tup2)] >= x_dict[subset_index, tup1[0], tup1[1]] + x_dict[subset_index, tup2[0], tup2[1]] - 1)
                mo.addConstr(var_y_dict[('y', subset_index, tup1, tup2)] <= x_dict[subset_index, tup1[0], tup1[1]])
                mo.addConstr(var_y_dict[('y', subset_index, tup1, tup2)] <= x_dict[subset_index, tup2[0], tup2[1]])
                

    # The cardinality of any subset from a partition P is from 0 to K
    # Equation B17
    for subset_index in range(num_clusters):
        z_constraint = []
        for catalog_index in range(num_catalog + 1):
            z_constraint.append(z_dict[subset_index, catalog_index])
        mo.addConstr(lhs = quicksum(variable for variable in z_constraint), sense=GRB.EQUAL, rhs = 1)
        
    # Only 1 of u^o_k is 1
    # Equation B18
    for subset_index in range(num_clusters):
        u_constraint = []
        for gridpoint_index in range(len(c_list)):
            u_constraint.append(u_dict[subset_index, gridpoint_index])
        mo.addConstr(lhs = quicksum(variable for variable in u_constraint), sense=GRB.EQUAL, rhs = 1)
        
    # Definition of variables chi
    # Equation B19
    for subset_index in range(num_clusters):
        chi_constraint_with_b = []
        chi_constraint = []
        x_constraint = []
        for breakpoint_index in range(1, num_breakpoints):
            chi_constraint_with_b.append(chi_dict[subset_index, breakpoint_index]*(np.exp(b_list[breakpoint_index])-np.exp(b_list[breakpoint_index-1])))
        for s,c in candidate_list:
            x_constraint.append(x_dict[subset_index, s, c]*kappa_dict[s,c])
        mo.addConstr(lhs = np.exp(b_list[0])* chi_dict[subset_index, 0] + quicksum(variable for variable in chi_constraint_with_b), 
                    sense=GRB.GREATER_EQUAL, 
                    rhs = quicksum(variable for variable in x_constraint))
        
        for breakpoint_index in range(num_breakpoints):
            chi_constraint.append(chi_dict[subset_index, breakpoint_index])
        for chi_index in range(len(chi_constraint) - 1):
            mo.addConstr(chi_constraint[chi_index] >= chi_constraint[chi_index + 1])
            
    # Definition of variables t
    # Equation B20
    product_of_cat_source_pairs_list = [prod for prod in list(itertools.combinations(candidate_list, r = 2)) if prod[0][1] != prod[1][1]]
        
    weighted_dist_dict = {}

    def get_distance_2(list_of_indexes, coord_dict):
        '''
        Given 2 pairs of (source_index, catalog_index), return the square distance between them
        df has the columns ('Catalog id', 'Source id', Coord 1, Coord 2)
        '''
        c1 = coord_dict[list_of_indexes[0]]
        c2 = coord_dict[list_of_indexes[1]]
        
        return ((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2)

    for p_tups in product_of_cat_source_pairs_list:
        weighted_dist_dict[p_tups] = kappa_dict[p_tups[0]]*kappa_dict[p_tups[1]] * get_distance_2(p_tups, coord_dict)

    for subset_index in range(num_clusters):
        for gridpoint_index in range(1,len(c_list)):
            mo.addConstr(lhs = t_list[subset_index], sense=GRB.GREATER_EQUAL,
                        rhs = -M * (1 - u_dict[subset_index, gridpoint_index])
                        + (1/(2*c_list[gridpoint_index])
                            * quicksum(var_y_dict[('y', subset_index, product_of_catalog_source_pairs[0], product_of_catalog_source_pairs[1])]
                                        * weighted_dist_dict[product_of_catalog_source_pairs]
                                        for product_of_catalog_source_pairs in product_of_cat_source_pairs_list)
                        ))

    # Definition of variables z
    # Equation B21
    for subset_index in range(num_clusters):
        x_constraint = []
        for s,c in candidate_list:
            x_constraint.append(x_dict[subset_index, s,c])
        for catalog_index in range(num_catalog + 1):
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.LESS_EQUAL,
                        rhs = catalog_index * z_dict[subset_index, catalog_index] + num_catalog * (1 - z_dict[subset_index, catalog_index]))
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.GREATER_EQUAL,
                        rhs = catalog_index * z_dict[subset_index, catalog_index])

    # Definition of variables u
    # Equation B222
    for subset_index in range(num_clusters):
        x_constraint = []
        for s,c in candidate_list:
            x_constraint.append(x_dict[subset_index, s,c]*round(kappa_dict[s,c],rounding_index))
        for gridpoint_index in range(len(c_list)):
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.LESS_EQUAL,
                        rhs = c_list[gridpoint_index] * u_dict[subset_index, gridpoint_index] + M_2 * (1 - u_dict[subset_index, gridpoint_index]))
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.GREATER_EQUAL,
                        rhs = c_list[gridpoint_index] * u_dict[subset_index, gridpoint_index])
        
    # Definition of variables p
    # Equation B23
    for subset_index in range(num_clusters):
        x_constraint = []
        for s,c in candidate_list:
            x_constraint.append(x_dict[subset_index, s,c])
        mo.addConstr(lhs = p_list[subset_index], sense=GRB.GREATER_EQUAL, 
                        rhs = np.log(2)*(1 - quicksum(variable for variable in x_constraint) - np.log(2)*z_dict[subset_index, 0]))
        
        
    # Break symmetry
    mo.addConstr(x_dict[0, candidate_list[0][0], candidate_list[0][1]] == 1.0)

    return mo, x_dict

def dirilp(data_df: pd.DataFrame, verbose = False):
    """Solve optimization problem using DirILP model.

    Args:
        data_df (pd.DataFrame): _description_
        max_clusters (int, optional): _description_. Defaults to -1.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    max_clusters = find_max_clusters(data_df) # uses quadratic formulation because smaller
    if verbose:
        print(f"Max Clusters using COP-KMeans: {max_clusters}")

    model, x = setup_dirilp(data_df, max_clusters, 0)

    if verbose:
        model.setParam("OutputFlag", 1)
    model.optimize()

    labels = []
    for (j, _, _), var in x.items():
        if var.X > 0.5:
            labels.append(j)
    assert len(labels) == data_df.shape[0] # sanity check that every point is assigned a label
    
    return pd.factorize(labels)[0] # factorize sets labels from 0 to max_labels