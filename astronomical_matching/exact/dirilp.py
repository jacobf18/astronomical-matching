import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum
from scipy.spatial.distance import pdist

from ..constants import ARCSEC_TO_RAD_2
from .miqcp import find_max_clusters


def setup_dirilp(data_df: pd.DataFrame, max_clusters=-1, min_clusters=0, verbose=False):
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
        coord_dict[source_image] = (
            row["coord1 (arcseconds)"],
            row["coord2 (arcseconds)"],
        )
        kappa_dict[source_image] = row["kappa"]

    C = np.log(ARCSEC_TO_RAD_2)
    # constant used for arcseconds to radians conversion

    sigma_min = data_df["Sigma"].min()
    sigma_max = data_df["Sigma"].max()

    mo = Model("likelihood")
    if not verbose:
        mo.setParam("OutputFlag", 0)

    M = np.ceil(
        1
        / sigma_min**4
        * sum(pdist(data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]))
        / (4 * 1 / sigma_max**2)
    )
    M_2 = num_catalog * max(kappa_dict.values())
    rounding_index = (
        -1
    )  # round number to nearest 10 if rounding_index = -1. to round to nearest 100, change to -2
    error_threshold = 1 / 100 * min(kappa_dict.values())

    t_list = []
    p_list = []
    b_list = [-2 * np.log(sigma_max)]
    c_list = [0, round(min(kappa_dict.values()), rounding_index)]
    # n = len(candidate_list)

    # Set Variables #
    # tup_to_ind = {v: i for i, v in enumerate(candidate_list)}

    # Compute b_list
    while b_list[-1] < np.log(num_catalog) - (2 * np.log(sigma_min)):
        b_list.append(b_list[-1] + error_threshold)

    num_breakpoints = len(b_list)  # = P in the paper

    # Compute c_list
    max_c = round(num_catalog * max(kappa_dict.values()), rounding_index)
    while c_list[-1] < max_c:
        c_list.append(c_list[-1] + 10 ** (-rounding_index))
        # c_list.append(c_list[-1]+(max_c/10**(-rounding_index)))

    # Variables for x
    x_dict = mo.addVars(num_clusters, candidate_list, vtype=GRB.BINARY)

    var_y_dict = {}
    # Variables for y
    for subset_index in range(num_clusters):
        for p1, p2 in list(itertools.combinations(candidate_list, r=2)):
            if p1[1] != p2[1]:
                var_y_dict[("y", subset_index, p1, p2)] = mo.addVar(
                    vtype=GRB.BINARY, name=str(("y", subset_index, p1, p2))
                )

    # Variables for z
    z_dict = mo.addVars(num_clusters, num_catalog + 1, vtype=GRB.BINARY)

    # Variables for t
    for subset_index in range(num_clusters):
        t_list.append(
            mo.addVar(lb=0, vtype=GRB.CONTINUOUS, name=str(("t", subset_index)))
        )

    # Variables for u
    u_dict = mo.addVars(num_clusters, len(c_list), vtype=GRB.BINARY)

    # Variables for p
    for subset_index in range(num_clusters):
        p_list.append(
            mo.addVar(
                lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(("p", subset_index))
            )
        )

    # Variables for chi
    chi_dict = mo.addVars(num_clusters, num_breakpoints, vtype=GRB.BINARY)

    # Variables to count number of sources in a cluster
    p = mo.addVars(num_clusters, lb=0, vtype=GRB.BINARY)

    # SET OBJECTIVES #

    # Set objective
    mo.setObjective(
        quicksum(
            p_list[subset_index]
            + b_list[0] * chi_dict[subset_index, 0]
            + error_threshold
            * quicksum(
                chi_dict[subset_index, breakpoint_index]
                for breakpoint_index in range(1, num_breakpoints)
            )
            + t_list[subset_index]
            for subset_index in range(num_clusters)
        )
        + (p.sum() * C),
        GRB.MINIMIZE,
    )

    # SET CONSTRAINTS #

    # All detections (i,c) needs to belong to some subset (S_j)
    # Equation B14
    # Each point assigned to a cluster
    for source, catalog in candidate_list:
        mo.addConstr(
            quicksum(x_dict[j, source, catalog] for j in range(num_clusters)) == 1
        )

    # |# objects|
    # p = 1 if there is a source in that cluster
    # p = 0 if no sources assigned to cluster
    for j in range(num_clusters):
        for s, c in candidate_list:
            mo.addConstr(p[j] >= x_dict[j, s, c])

    mo.addConstr(p.sum() >= min_clusters)

    # Every subset takes no more than 1 detection from each catalog
    # # Equation B15

    # Each cluster has at most one source from a catalog
    sources_by_catalog = defaultdict(list)
    for source, catalog in candidate_list:
        sources_by_catalog[catalog].append(source)

    for j, c in itertools.product(range(num_clusters), range(num_catalog)):
        mo.addConstr(
            quicksum(x_dict[j, source, c] for source in sources_by_catalog[c]) <= 1
        )

    # Definition of variables y
    # Equation B16
    for subset_index in range(num_clusters):
        for tup1, tup2 in list(itertools.combinations(candidate_list, r=2)):
            if tup1[1] != tup2[1]:
                mo.addConstr(
                    var_y_dict[("y", subset_index, tup1, tup2)]
                    >= x_dict[subset_index, tup1[0], tup1[1]]
                    + x_dict[subset_index, tup2[0], tup2[1]]
                    - 1
                )
                mo.addConstr(
                    var_y_dict[("y", subset_index, tup1, tup2)]
                    <= x_dict[subset_index, tup1[0], tup1[1]]
                )
                mo.addConstr(
                    var_y_dict[("y", subset_index, tup1, tup2)]
                    <= x_dict[subset_index, tup2[0], tup2[1]]
                )

    # The cardinality of any subset from a partition P is from 0 to K
    # Equation B17
    for subset_index in range(num_clusters):
        z_constraint = []
        for catalog_index in range(num_catalog + 1):
            z_constraint.append(z_dict[subset_index, catalog_index])
        mo.addConstr(
            lhs=quicksum(variable for variable in z_constraint), sense=GRB.EQUAL, rhs=1
        )

    # Only 1 of u^o_k is 1
    # Equation B18
    for subset_index in range(num_clusters):
        u_constraint = []
        for gridpoint_index in range(len(c_list)):
            u_constraint.append(u_dict[subset_index, gridpoint_index])
        mo.addConstr(
            lhs=quicksum(variable for variable in u_constraint), sense=GRB.EQUAL, rhs=1
        )

    # Definition of variables chi
    # Equation B19
    for subset_index in range(num_clusters):
        chi_constraint_with_b = []
        chi_constraint = []
        x_constraint = []
        for breakpoint_index in range(1, num_breakpoints):
            chi_constraint_with_b.append(
                chi_dict[subset_index, breakpoint_index]
                * (
                    np.exp(b_list[breakpoint_index])
                    - np.exp(b_list[breakpoint_index - 1])
                )
            )
        for s, c in candidate_list:
            x_constraint.append(x_dict[subset_index, s, c] * kappa_dict[s, c])
        mo.addConstr(
            lhs=np.exp(b_list[0]) * chi_dict[subset_index, 0]
            + quicksum(variable for variable in chi_constraint_with_b),
            sense=GRB.GREATER_EQUAL,
            rhs=quicksum(variable for variable in x_constraint),
        )

        for breakpoint_index in range(num_breakpoints):
            chi_constraint.append(chi_dict[subset_index, breakpoint_index])
        for chi_index in range(len(chi_constraint) - 1):
            mo.addConstr(chi_constraint[chi_index] >= chi_constraint[chi_index + 1])

    # Definition of variables t
    # Equation B20
    product_of_cat_source_pairs_list = [
        prod
        for prod in list(itertools.combinations(candidate_list, r=2))
        if prod[0][1] != prod[1][1]
    ]

    weighted_dist_dict = {}

    def get_distance_2(list_of_indexes, coord_dict):
        """
        Given 2 pairs of (source_index, catalog_index), return the square distance between them
        df has the columns ('Catalog id', 'Source id', Coord 1, Coord 2)
        """
        c1 = coord_dict[list_of_indexes[0]]
        c2 = coord_dict[list_of_indexes[1]]

        return ((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2)

    for p_tups in product_of_cat_source_pairs_list:
        weighted_dist_dict[p_tups] = (
            kappa_dict[p_tups[0]]
            * kappa_dict[p_tups[1]]
            * get_distance_2(p_tups, coord_dict)
        )

    for subset_index in range(num_clusters):
        for gridpoint_index in range(1, len(c_list)):
            mo.addConstr(
                lhs=t_list[subset_index],
                sense=GRB.GREATER_EQUAL,
                rhs=-M * (1 - u_dict[subset_index, gridpoint_index])
                + (
                    1
                    / (2 * c_list[gridpoint_index])
                    * quicksum(
                        var_y_dict[
                            (
                                "y",
                                subset_index,
                                product_of_catalog_source_pairs[0],
                                product_of_catalog_source_pairs[1],
                            )
                        ]
                        * weighted_dist_dict[product_of_catalog_source_pairs]
                        for product_of_catalog_source_pairs in product_of_cat_source_pairs_list
                    )
                ),
            )

    # Definition of variables z
    # Equation B21
    for subset_index in range(num_clusters):
        x_constraint = []
        for s, c in candidate_list:
            x_constraint.append(x_dict[subset_index, s, c])
        for catalog_index in range(num_catalog + 1):
            mo.addConstr(
                lhs=quicksum(variable for variable in x_constraint),
                sense=GRB.LESS_EQUAL,
                rhs=catalog_index * z_dict[subset_index, catalog_index]
                + num_catalog * (1 - z_dict[subset_index, catalog_index]),
            )
            mo.addConstr(
                lhs=quicksum(variable for variable in x_constraint),
                sense=GRB.GREATER_EQUAL,
                rhs=catalog_index * z_dict[subset_index, catalog_index],
            )

    # Definition of variables u
    # Equation B222
    for subset_index in range(num_clusters):
        x_constraint = []
        for s, c in candidate_list:
            x_constraint.append(
                x_dict[subset_index, s, c] * round(kappa_dict[s, c], rounding_index)
            )
        for gridpoint_index in range(len(c_list)):
            mo.addConstr(
                lhs=quicksum(variable for variable in x_constraint),
                sense=GRB.LESS_EQUAL,
                rhs=c_list[gridpoint_index] * u_dict[subset_index, gridpoint_index]
                + M_2 * (1 - u_dict[subset_index, gridpoint_index]),
            )
            mo.addConstr(
                lhs=quicksum(variable for variable in x_constraint),
                sense=GRB.GREATER_EQUAL,
                rhs=c_list[gridpoint_index] * u_dict[subset_index, gridpoint_index],
            )

    # Definition of variables p
    # Equation B23
    for subset_index in range(num_clusters):
        x_constraint = []
        for s, c in candidate_list:
            x_constraint.append(x_dict[subset_index, s, c])
        mo.addConstr(
            lhs=p_list[subset_index],
            sense=GRB.GREATER_EQUAL,
            rhs=np.log(2)
            * (
                1
                - quicksum(variable for variable in x_constraint)
                - np.log(2) * z_dict[subset_index, 0]
            ),
        )

    # Break symmetry
    mo.addConstr(x_dict[0, candidate_list[0][0], candidate_list[0][1]] == 1.0)

    return mo, x_dict


def dirilp(data_df: pd.DataFrame, verbose=False):
    """Solve optimization problem using DirILP model.

    Args:
        data_df (pd.DataFrame): _description_
        max_clusters (int, optional): _description_. Defaults to -1.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # uses quadratic formulation because it is faster to setup
    max_clusters = find_max_clusters(data_df)
    
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
    assert (
        len(labels) == data_df.shape[0]
    )  # sanity check that every point is assigned a label

    return pd.factorize(labels)[0]  # factorize sets labels from 0 to max_labels
