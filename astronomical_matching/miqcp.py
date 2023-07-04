import itertools
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum  # type: ignore
from scipy.spatial.distance import pdist  # type: ignore

from .constants import ARCSEC_TO_RAD_2
from .cop_kmeans import run_cop_kmeans
from .utils import sterling2


def setup_miqcp_model(data_df, max_clusters=-1, min_clusters=0, verbose=False):
    num_datapoints = data_df.shape[0]
    num_clusters = max_clusters
    if max_clusters == -1:
        num_clusters = num_datapoints
    num_catalogs = data_df["ImageID"].unique().shape[0]
    dims = 2

    # constant used for arcseconds to radians conversion
    C = math.log(ARCSEC_TO_RAD_2)

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
        coord_dict[source_image] = (
            row["coord1 (arcseconds)"],
            row["coord2 (arcseconds)"],
        )
        kappa_dict[source_image] = row["kappa"]

    # Add cluster variables
    cluster_vars = model.addVars(
        num_clusters, dims, lb=-float("inf"), ub=float("inf")
    )

    # Add boolean variables (cluster-sources)
    x = model.addVars(
        candidate_list, list(range(num_clusters)), vtype=GRB.BINARY, name="x"
    )

    for (source, catalog, k), var in x.items():
        var.setAttr("BranchPriority", 1)

    # Number of clusters
    p = model.addVars(num_clusters, lb=0, vtype=GRB.BINARY)

    # Add M variable
    M = (
        np.max(pdist(data_df[["coord1 (arcseconds)", "coord2 (arcseconds)"]]))
        * data_df["kappa"].max()
        * 1.1
    )
    # M = 10**6

    # Add max cluster distance variables
    r_dict = model.addVars(candidate_list, lb=0.0, ub=float("inf"))

    # Log term
    error_threshold = 1 / 10 * np.log(data_df["kappa"].min())

    var_chi_dict = {}
    sigma_max = data_df["Sigma"].max()
    sigma_min = data_df["Sigma"].min()

    # b_list = [np.log(1/(sigma_max)**2) + C]
    # b_list = [np.log(1/(sigma_max)**2)]
    b_list = [(-2) * np.log(sigma_max)]
    # Compute b_list
    # while b_list[-1] < np.log(num_catalogs) - np.log((sigma_min)**2) + C:
    while b_list[-1] < np.log(num_catalogs) - (2 * np.log((sigma_min))):
        b_list.append(b_list[-1] + error_threshold)

    num_breakpoints = len(b_list)  # = P in the paper

    # Variables for chi
    for j in range(num_clusters):
        for b_i in range(num_breakpoints):
            var_chi_dict[("chi", j, b_i)] = model.addVar(
                vtype=GRB.BINARY, name=str(("chi", j, b_i))
            )

    s = model.addVars(num_clusters, lb=0, vtype=GRB.INTEGER)

    # Add Sterling number variables
    sterling_vars = model.addVars(range(1, num_clusters + 1), lb=0)
    z = model.addVars(range(1, num_clusters + 1), lb=0, vtype=GRB.BINARY)

    # Objective #
    sum_ln_kappa_rad = (C * num_datapoints) + np.log(data_df["kappa"]).sum()
    model.setObjective(
        (0.5 * r_dict.sum())
        + (np.log(2) * p.sum())
        - (np.log(2) * s.sum())
        + quicksum(
            (b_list[0] * var_chi_dict[("chi", j, 0)])
            + (
                error_threshold
                * quicksum(
                    var_chi_dict[("chi", j, b_i)]
                    for b_i in range(1, num_breakpoints)
                )
            )
            for j in range(num_clusters)
        )
        + (p.sum() * C)
        - sum_ln_kappa_rad
        + sterling_vars.sum(),
        GRB.MINIMIZE,
    )

    # Constraints #
    # Each point assigned to a cluster
    for source, catalog in candidate_list:
        model.addConstr(
            quicksum(x[(source, catalog, j)] for j in range(num_clusters)) == 1
        )

    # |# objects|
    # p = 1 if there is a source in that cluster
    # p = 0 if no sources assigned to cluster
    for j in range(num_clusters):
        for source, catalog in candidate_list:
            model.addConstr(p[j] >= x[source, catalog, j])

    # lower bound on objects for getting a maximum object count
    model.addConstr(p.sum() >= min_clusters)

    # |# sources| * ln(2)
    for j in range(num_clusters):
        model.addConstr(
            s[j]
            == quicksum(
                x[source, catalog, j] for source, catalog in candidate_list
            )
        )

    # Each cluster has at most one source from a catalog
    sources_by_catalog = defaultdict(list)
    for source, catalog in candidate_list:
        sources_by_catalog[catalog].append(source)

    for j, c in itertools.product(range(num_clusters), range(num_catalogs)):
        model.addConstr(
            quicksum(x[(source, c, j)] for source in sources_by_catalog[c]) <= 1
        )

    # Min and max for cluster variables
    # Get coordinates
    x_coords = data_df["coord1 (arcseconds)"]
    y_coords = data_df["coord2 (arcseconds)"]

    for j in range(num_clusters):
        model.addConstr(cluster_vars[j, 0] == [min(x_coords), max(x_coords)])
        model.addConstr(cluster_vars[j, 1] == [min(y_coords), max(y_coords)])

    # Break symmetry
    first_s, first_c = candidate_list[0]
    model.addConstr(x[(first_s, first_c, 0)] == 1)

    # Big-M constraints
    for (source, catalog), coord in coord_dict.items():
        for j in range(num_clusters):
            model.addQConstr(
                (
                    kappa_dict[(source, catalog)]
                    * (  # in arcseconds^-2
                        (
                            (cluster_vars[j, 0] - coord[0])
                            * (cluster_vars[j, 0] - coord[0])
                        )
                        + (
                            (cluster_vars[j, 1] - coord[1])
                            * (cluster_vars[j, 1] - coord[1])
                        )
                    )
                )  # in arcseconds ^ 2
                <= r_dict[(source, catalog)]
                + (M * (1 - x[(source, catalog, j)]))
            )

    # Sterling number vars
    M2 = math.log(sterling2(num_datapoints, num_clusters)) * 2

    model.addConstr(z.sum() == 1)

    for j in range(1, num_clusters + 1):
        model.addConstr(p.sum() <= j * z[j] + num_clusters * (1 - z[j]))
        model.addConstr(p.sum() >= j * z[j])

        model.addConstr(
            sterling_vars[j] - math.log(sterling2(num_datapoints, j))
            <= M2 * (1 - z[j])
        )
        model.addConstr(
            sterling_vars[j] - math.log(sterling2(num_datapoints, j))
            >= -M2 * (1 - z[j])
        )

    # Definition of variables chi
    # Equation B19
    for j in range(num_clusters):
        chi_constraint_with_b = []
        chi_constraint = []
        x_constraint = []
        for breakpoint_index in range(1, num_breakpoints):
            chi_constraint_with_b.append(
                var_chi_dict[("chi", j, breakpoint_index)]
                * (
                    np.exp(b_list[breakpoint_index])
                    - np.exp(b_list[breakpoint_index - 1])
                )
            )
        for source, catalog in candidate_list:
            x_constraint.append(
                x[(source, catalog, j)] * kappa_dict[(source, catalog)]
            )
        model.addConstr(
            np.exp(b_list[0]) * var_chi_dict[("chi", j, 0)]
            + quicksum(variable for variable in chi_constraint_with_b)
            >= quicksum(variable for variable in x_constraint)
        )

        for breakpoint_index in range(num_breakpoints):
            chi_constraint.append(var_chi_dict[("chi", j, breakpoint_index)])
        for chi_index in range(len(chi_constraint) - 1):
            model.addConstr(
                chi_constraint[chi_index] >= chi_constraint[chi_index + 1]
            )

    model.setParam("NumericFocus", 2)  # for numerical stability
    model.setParam("NodefileStart", 2.0)

    return model, x


def find_max_clusters(data_df) -> int:
    """Find the maximum number of clusters by bounding
    and relaxing the model and comparing
    the best possible bounded bayes factor to the
    feasible point returned by constrained
    kmeans.

    Args:
        data_df (pd.DataFrame): dataframe of coordinates and uncertainties

    Returns:
        int: maximum number of clusters
    """
    _, _, cop_kmeans_bayes = run_cop_kmeans(
        data_df, min_k=1, max_k=data_df.shape[0]
    )
    max_cluster = 0
    for c in range(data_df.shape[0]):
        model, _ = setup_miqcp_model(data_df, max_clusters=-1, min_clusters=c)

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


def miqcp(
    data_df: pd.DataFrame, verbose=False, preDual=False, preQLinearize=False
):
    max_clusters = find_max_clusters(data_df)
    # max_clusters = 2
    if verbose:
        print(f"Max Clusters using COP-KMeans: {max_clusters}")

    model, x = setup_miqcp_model(data_df, max_clusters, 0)

    # Solve MIQCP
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
    assert (
        len(labels) == data_df.shape[0]
    )  # sanity check that every point is assigned a label

    # factorize sets labels from 0 to max_labels
    return pd.factorize(labels)[0]
