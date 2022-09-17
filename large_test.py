import random
import string
import math
import time
import itertools
from math import log as ln
from math import exp
import pandas as pd
import numpy as np
from gurobipy import *
from scipy import spatial
from scipy import sparse
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

num_catalog = 40


def mock(num=10, size=10, seed=None):
    "return dataframe of simulated objects"
    if seed is not None:
        np.random.seed(seed)
    obj = np.random.random(size=(num, 3))
    obj[:, 0:2] *= size
    obj[:, 1] -= size / 2  # The location is scaled to simulate objects on a 10x10 grid.
    # objid = np.arange(obj.shape[0]) # object id
    df = pd.DataFrame(obj, columns=["x", "y", "u"])
    return df


def cat(mdf):
    "return simulated catalog and selection"
    sigma = 1 / 100 * np.random.randint(4, 10, (mdf.shape[0], 1))
    det = mdf.values[:, :2] + sigma * np.random.randn(mdf.shape[0], 2)
    df = pd.DataFrame(det, columns=["x", "y"])
    df["Sigma"] = sigma
    a, b = (0, 1)
    sel = np.logical_and(mdf.u >= a, mdf.u < b)
    # df['selected'] = np.array(sel, dtype=np.bool)
    return df.values, sel


def random_cat(mdf):
    "return simulated catalog with random selection intervals"
    sigma = 1 / 100 * np.random.randint(4, 10, (mdf.shape[0], 1))
    det = mdf.values[:, :2] + sigma * np.random.randn(mdf.shape[0], 2)
    df = pd.DataFrame(det, columns=["x", "y"])
    df["Sigma"] = sigma
    num_endpoints = random.randrange(
        6, 10, 2
    )  # Random multiple of 2 from 6 to 16 #So the number of intervals = num_endpoints / 2
    endpoints_list = sorted([random.random() for x in range(num_endpoints)])
    selection_boolean_list = []
    for index in range(int(num_endpoints / 2)):
        a = endpoints_list[2 * index]
        b = endpoints_list[2 * index + 1]
        sel = np.logical_and(mdf.u >= a, mdf.u < b)
        selection_boolean_list.append(sel)
    final_sel = selection_boolean_list[0]
    for i in range(1, int(num_endpoints / 2)):
        final_sel = final_sel | selection_boolean_list[i]
    df = df[final_sel].values
    return df, final_sel


np.random.seed(42)
num = 100
size = 30
Qmat = np.empty((1, 2))
m = mock(num=num, size=size, seed=20)
# generate 3 catalogs with sigma=0.1" and different selection intevals
cmat = []
sel_list = []
for i in range(num_catalog):  # List of catalogs with coordinates and conditions
    catalog_i, final_sel = cat(m)
    cmat.append(catalog_i)
    sel_list.append(final_sel)

true_matching_dict = {}
count_list = [0] * num
for source_index in range(num):
    index_list = []
    for catalog_index in range(num_catalog):
        if sel_list[catalog_index][source_index] == True:
            index_list.append((catalog_index, count_list[catalog_index]))
            count_list[catalog_index] += 1
    true_matching_dict[source_index] = index_list

label = []
X = []
for i in range(num_catalog):
    label += [i] * len(cmat[i])
    first_coord, second_coord = cmat[i][:, :2].T
    X += [(i, j) for i, j in zip(first_coord, second_coord)]
label = np.array(label)
X = np.array(X)
sigma_max = 0.1
result = DBSCAN(eps=5 * sigma_max, min_samples=2).fit(X)
y_pred = DBSCAN(eps=5 * sigma_max, min_samples=2).fit_predict(X)


def get_candidate_list(clustering_result, catalog_label, clustering_label):
    candidate_dict = {}
    list_of_label = catalog_label[
        np.where(clustering_result.labels_ == clustering_label)
    ]  # The catalog id of all sources found in the island
    for j in np.unique(list_of_label):
        list_of_indices = []
        for k in range(sum(list_of_label == j)):
            list_of_indices.append(
                (
                    (
                        np.where(
                            cmat[j][:, :2]
                            == clustering_result.components_[
                                np.where(clustering_result.labels_ == clustering_label)
                            ][list_of_label == j][k]
                        )[0][0]
                    ),
                    j,
                )
            )
        candidate_dict[j] = list_of_indices  # catalog id: (source id, catalog id)

    # Get all the (catalog_id, source_id) pairs
    candidate_list = []
    for key in candidate_dict.keys():
        list_of_value = candidate_dict[key]
        for pair in list_of_value:
            candidate_list.append(pair)

    catalog_list = []
    source_list = []
    coord_1_list = []
    coord_2_list = []
    Sigma_list = []
    for catalog_source_pair in candidate_list:
        catalog_list.append(catalog_source_pair[1])
        source_list.append(catalog_source_pair[0])
        coord_1_list.append(cmat[catalog_source_pair[1]][catalog_source_pair[0]][0])
        coord_2_list.append(cmat[catalog_source_pair[1]][catalog_source_pair[0]][1])
        Sigma_list.append(cmat[catalog_source_pair[1]][catalog_source_pair[0]][2])
    df = pd.DataFrame(
        {
            "Catalog_id": catalog_list,
            "Source_id": source_list,
            "Coordinate_1": coord_1_list,
            "Coordinate_2": coord_2_list,
        }
    )  # Make a dataframe from the information provided
    return (candidate_list, df, catalog_list, Sigma_list)


def get_distance(list_of_indexes, df):
    """
    Given 2 pairs of (source_index, catalog_index), return the square distance between them
    df has the columns ('Catalog id', 'Source id', Coord 1, Coord 2)
    """
    coord_list = []
    for i in range(2):
        coord_list += [
            np.array(
                df[
                    (df["Catalog_id"] == list_of_indexes[i][1])
                    & (df["Source_id"] == list_of_indexes[i][0])
                ].iloc[:, -2:]
            )[0]
        ]
    array_matrix = np.array(coord_list)
    return np.linalg.norm(array_matrix[1] - array_matrix[0]) ** 2


def get_distance_2(list_of_indexes, temp_dict):
    """
    Given 2 pairs of (source_index, catalog_index), return the square distance between them
    df has the columns ('Catalog id', 'Source id', Coord 1, Coord 2)
    """
    c1 = temp_dict[list_of_indexes[0]]
    c2 = temp_dict[list_of_indexes[1]]

    return ((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2)


def sum_of_distance(list_of_objects, df):
    """
    Given n pairs of (catalog_index, source_index), return the sum of all pairwise square distance.
    """
    num_of_objects = len(list_of_objects)
    coord_list = []
    for i in range(num_of_objects):
        coord_list += [
            np.array(
                df[
                    (df["Catalog_id"] == list_of_objects[i][1])
                    & (df["Source_id"] == list_of_objects[i][0])
                ].iloc[:, -2:]
            )[0]
        ]
    array_matrix = np.array(coord_list)
    pairwise_dist = spatial.distance.pdist(np.array(array_matrix)) ** 2
    sum_of_square_dist = sum(pairwise_dist)
    return sum_of_square_dist


def Bayes_factor(list_of_objects, df):
    """
    Compute -ln B_o
    """
    sum_ln_kappa_rad = 0
    kappa_rad_sum = 0
    kappa_sum = 0
    neg_ln_Bayes = 0
    num_of_objects = len(list_of_objects)
    for object in list_of_objects:
        sum_ln_kappa_rad += ln(kappa_rad_dict[object])
        kappa_rad_sum += kappa_rad_dict[object]
        kappa_sum += kappa_dict[object]
    for index_1 in range(num_of_objects):
        for index_2 in range(index_1 + 1, num_of_objects):
            neg_ln_Bayes += (
                (1 / (4 * kappa_sum))
                * kappa_dict[list_of_objects[index_1]]
                * kappa_dict[list_of_objects[index_2]]
                * get_distance([list_of_objects[index_1], list_of_objects[index_2]], df)
            )
    neg_ln_Bayes = (
        neg_ln_Bayes
        + (1 - num_of_objects) * ln(2)
        - sum_ln_kappa_rad
        + ln(kappa_rad_sum)
    )
    return neg_ln_Bayes


def compute_distance_dictionary(list_of_indices, df):
    """
    Return a dictionary with the form: dict[('Source_id_1', 'Catalog_id_1'), ('Source_id_2', 'Catalog_id_2')] = square distance between them.
    """
    distance_dict = {}
    for current_pair_index in range(len(list_of_indices)):
        for next_pair_index in range(current_pair_index + 1, len(list_of_indices)):
            if (
                list_of_indices[next_pair_index][1]
                != list_of_indices[current_pair_index][1]
            ):  # Only find distances for sources from different catalogs
                distance_dict[
                    (
                        list_of_indices[current_pair_index],
                        list_of_indices[next_pair_index],
                    )
                ] = get_distance(
                    [
                        list_of_indices[current_pair_index],
                        list_of_indices[next_pair_index],
                    ],
                    df,
                )
    return distance_dict


def compute_distance_dictionary_2(list_of_indices, temp_dict):
    """
    Return a dictionary with the form: dict[('Source_id_1', 'Catalog_id_1'), ('Source_id_2', 'Catalog_id_2')] = square distance between them.
    """
    distance_dict = {}
    for current_pair_index in range(len(list_of_indices)):
        for next_pair_index in range(current_pair_index + 1, len(list_of_indices)):
            if (
                list_of_indices[next_pair_index][1]
                != list_of_indices[current_pair_index][1]
            ):  # Only find distances for sources from different catalogs
                distance_dict[
                    (
                        list_of_indices[current_pair_index],
                        list_of_indices[next_pair_index],
                    )
                ] = get_distance_2(
                    [
                        list_of_indices[current_pair_index],
                        list_of_indices[next_pair_index],
                    ],
                    temp_dict,
                )
    return distance_dict


def mycallback(model, where):
    start_time = [0]
    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(
            GRB.Callback.PRE_COLDEL
        )  # number of cols removed by presolve to this point
        rdels = model.cbGet(
            GRB.Callback.PRE_ROWDEL
        )  # number of rows removed by presolve to this point
    elif where == GRB.Callback.SIMPLEX:  # Currently in simplex
        # Simplex callback
        itcnt = model.cbGet(GRB.Callback.SPX_ITRCNT)  # Current simplex iteration count
        # if itcnt - model._lastiter >= 100:
        model._lastiter = itcnt
        obj = model.cbGet(GRB.Callback.SPX_OBJVAL)  # Current simplex objective value
        ispert = model.cbGet(GRB.Callback.SPX_ISPERT)
        pinf = model.cbGet(GRB.Callback.SPX_PRIMINF)  # Current primal infeasibility
        dinf = model.cbGet(GRB.Callback.SPX_DUALINF)  # Current dual infeasibility
        if ispert == 0:
            ch = " "
        elif ispert == 1:
            ch = "S"
        else:
            ch = "P"
    elif where == GRB.Callback.MIP:  # Currently in MIP
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)  # Current explored node count
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)  # Current best objective
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)  # Current best objective bound
        solcnt = model.cbGet(
            GRB.Callback.MIP_SOLCNT
        )  # Current count of feasible solutions found
        # if nodecnt - model._lastnode >= 100:
        model._lastnode = nodecnt
        actnodes = model.cbGet(GRB.Callback.MIP_NODLFT)  # Current unexplored node count
        itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)  # Current simplex iteration count
        cutcnt = model.cbGet(
            GRB.Callback.MIP_CUTCNT
        )  # Current count of cutting planes applied
    elif where == GRB.Callback.MIPSOL:  # Found a new MIP incumbent
        # MIP solution callback
        start_time.append(model.cbGet(GRB.Callback.RUNTIME))
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)  # Current explored node count
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)  # Objective value for new solution
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        solcnt = model.cbGet(
            GRB.Callback.MIPSOL_SOLCNT
        )  # Curent count of feasible solutions found
        best_gap = abs(100 * (objbnd - objbst) / objbst)
        current_gap = abs(100 * (objbnd - obj) / obj)  # Gap of the new solution found
        x_MIP = model.cbGetSolution(model._vars)
        x_MIP = np.array(x_MIP)
        index_list = np.where(x_MIP > 0.1)
        res_list = [model._vars[i] for i in index_list[0]]
        nonzero_x = []
        for solution in res_list:
            if solution.Varname[1:4] == "'x'":
                nonzero_x.append(solution.VarName)
        for i in range(len(nonzero_x)):
            print("\n x = %s" % nonzero_x[i])
        model._logfile.write(
            "\n**** New solution # %d, Obj %g, Current Gap %g%%, Best Obj %g, Best Gap %g%%, Elapsed time %g,"
            % (
                solcnt,
                obj,
                current_gap,
                objbst,
                best_gap,
                model.cbGet(GRB.Callback.RUNTIME),
            )
        )
        for i in range(len(nonzero_x)):
            model._logfile.write("\n x = %s" % nonzero_x[i])

    elif where == GRB.Callback.MIPNODE:  # Currently exploring a MIP node
        # MIP node callback
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(
                model._vars
            )  # Values from the node relaxation solution at the current node
            model.cbSetSolution(model.getVars(), x)
    elif where == GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)

    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        model._logfile.write(msg)


# Get a particular island to run the matching procedure
candidate_list, df, catalog_list, Sigma_list = get_candidate_list(result, label, 43)

temp_dict = {
    (row[1], row[0]): (row[2], row[3]) for row in df.itertuples(index=False, name=None)
}

detection_total = len(
    candidate_list
)  # Total number of sources across different catalogs

kappa_dict = {}
kappa_rad_dict = {}
ln_kappa_rad_dict = {}

for i in range(len(candidate_list)):
    kappa_dict[(candidate_list[i][0], candidate_list[i][1])] = (
        1 / (Sigma_list[i]) ** 2
    )  # in arcsecond
    kappa_rad_dict[(candidate_list[i][0], candidate_list[i][1])] = 1 / (
        (Sigma_list[i] * np.pi / 180 / 3600) ** 2
    )  # in radian
    ln_kappa_rad_dict[(candidate_list[i][0], candidate_list[i][1])] = ln(
        1 / ((Sigma_list[i] * np.pi / 180 / 3600) ** 2)
    )

sigma_min = min(Sigma_list)
sigma_max = max(Sigma_list)

######################### ILP Formulation ###########################
import time

t1 = time.perf_counter()

mo = Model("likelihood")

M = np.ceil(
    1 / sigma_min**4 * sum_of_distance(candidate_list, df) / (4 * 1 / sigma_max**2)
)
M_2 = num_catalog * max(kappa_dict.values())
rounding_index = (
    -2
)  # round number to nearest 10 if rounding_index = -1. to round to nearest 100, change to -2
error_threshold = 1 / 100 * min(ln_kappa_rad_dict.values())

var_x_dict = {}
var_y_dict = {}
var_z_dict = {}
var_u_dict = {}
var_chi_dict = {}
t_list = []
p_list = []
b_list = [ln(1 / (sigma_max * np.pi / 180 / 3600) ** 2)]
c_list = [0, round(min(kappa_dict.values()), rounding_index)]
########################### SET VARIABLES ###########################

# Compute b_list
while b_list[-1] < ln(num_catalog / (sigma_min * np.pi / 180 / 3600) ** 2):
    b_list.append(b_list[-1] + error_threshold)

num_breakpoints = len(b_list)  # = P in the paper

# Compute c_list
while c_list[-1] < round(num_catalog * max(kappa_dict.values()), rounding_index):
    c_list.append(c_list[-1] + 10 ** (-rounding_index))

# Variables for x
for subset_index in range(detection_total):
    for catalog_source_pair in candidate_list:
        var_x_dict[("x", subset_index, catalog_source_pair)] = mo.addVar(
            vtype=GRB.BINARY, name=str(("x", subset_index, catalog_source_pair))
        )

# Variables for y
for subset_index in range(detection_total):
    for product_of_catalog_source_pairs in list(
        itertools.combinations(candidate_list, r=2)
    ):
        if (
            product_of_catalog_source_pairs[0][1]
            != product_of_catalog_source_pairs[1][1]
        ):
            var_y_dict[
                (
                    "y",
                    subset_index,
                    product_of_catalog_source_pairs[0],
                    product_of_catalog_source_pairs[1],
                )
            ] = mo.addVar(
                vtype=GRB.BINARY,
                name=str(
                    (
                        "y",
                        subset_index,
                        product_of_catalog_source_pairs[0],
                        product_of_catalog_source_pairs[1],
                    )
                ),
            )

# Variables for z
for subset_index in range(detection_total):
    for catalog_index in range(num_catalog + 1):
        var_z_dict[("z", subset_index, catalog_index)] = mo.addVar(
            vtype=GRB.BINARY, name=str(("z", subset_index, catalog_index))
        )

# Variables for t
for subset_index in range(detection_total):
    t_list.append(mo.addVar(lb=0, vtype=GRB.CONTINUOUS, name=str(("t", subset_index))))

# Variables for u
for subset_index in range(detection_total):
    for gridpoint_index in range(len(c_list)):
        var_u_dict[("u", subset_index, gridpoint_index)] = mo.addVar(
            vtype=GRB.BINARY, name=str(("u", subset_index, gridpoint_index))
        )

# Variables for p
for subset_index in range(detection_total):
    p_list.append(
        mo.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(("p", subset_index)))
    )

# Variables for chi
for subset_index in range(detection_total):
    for breakpoint_index in range(num_breakpoints):
        var_chi_dict[("chi", subset_index, breakpoint_index)] = mo.addVar(
            vtype=GRB.BINARY, name=str(("chi", subset_index, breakpoint_index))
        )

########################### SET OBJECTIVES ###########################

# Set objective
mo.setObjective(
    quicksum(
        p_list[subset_index]
        - quicksum(
            var_x_dict["x", subset_index, catalog_source_pair]
            * ln_kappa_rad_dict[catalog_source_pair]
            for catalog_source_pair in candidate_list
        )
        + b_list[0] * var_chi_dict[("chi", subset_index, 0)]
        + error_threshold
        * quicksum(
            var_chi_dict[("chi", subset_index, breakpoint_index)]
            for breakpoint_index in range(1, num_breakpoints)
        )
        + t_list[subset_index]
        for subset_index in range(detection_total)
    ),
    GRB.MINIMIZE,
)

########################### SET CONSTRAINTS ###########################

# All detections (i,c) needs to belong to some subset (S_j)
# Equation B14
for catalog_source_pair in candidate_list:
    x_constraint = []
    for variable in var_x_dict.keys():
        if variable[-1] == catalog_source_pair:
            x_constraint.append(var_x_dict[variable])
    mo.addConstr(
        lhs=quicksum(variable for variable in x_constraint), sense=GRB.EQUAL, rhs=1
    )

# Every subset takes no more than 1 detection from each catalog
# Equation B15
for subset_index in range(detection_total):
    for catalog_index in catalog_list:
        x_constraint = []
        for variable in var_x_dict.keys():
            if (variable[1] == subset_index) & (variable[-1][1] == catalog_index):
                x_constraint.append(var_x_dict[variable])
        mo.addConstr(
            lhs=quicksum(variable for variable in x_constraint),
            sense=GRB.LESS_EQUAL,
            rhs=1,
        )

# Definition of variables y
# Equation B16
for subset_index in range(detection_total):
    for product_of_catalog_source_pairs in list(
        itertools.combinations(candidate_list, r=2)
    ):
        if (
            product_of_catalog_source_pairs[0][1]
            != product_of_catalog_source_pairs[1][1]
        ):
            mo.addConstr(
                var_y_dict[
                    (
                        "y",
                        subset_index,
                        product_of_catalog_source_pairs[0],
                        product_of_catalog_source_pairs[1],
                    )
                ]
                >= var_x_dict[("x", subset_index, product_of_catalog_source_pairs[0])]
                + var_x_dict[("x", subset_index, product_of_catalog_source_pairs[1])]
                - 1
            )
            mo.addConstr(
                var_y_dict[
                    (
                        "y",
                        subset_index,
                        product_of_catalog_source_pairs[0],
                        product_of_catalog_source_pairs[1],
                    )
                ]
                <= var_x_dict[("x", subset_index, product_of_catalog_source_pairs[0])]
            )
            mo.addConstr(
                var_y_dict[
                    (
                        "y",
                        subset_index,
                        product_of_catalog_source_pairs[0],
                        product_of_catalog_source_pairs[1],
                    )
                ]
                <= var_x_dict[("x", subset_index, product_of_catalog_source_pairs[1])]
            )

# The cardinality of any subset from a partition P is from 0 to K
# Equation B17
for subset_index in range(detection_total):
    z_constraint = []
    for catalog_index in range(num_catalog + 1):
        z_constraint.append(var_z_dict[("z", subset_index, catalog_index)])
    mo.addConstr(
        lhs=quicksum(variable for variable in z_constraint), sense=GRB.EQUAL, rhs=1
    )

# Only 1 of u^o_k is 1
# Equation B18
for subset_index in range(detection_total):
    u_constraint = []
    for gridpoint_index in range(len(c_list)):
        u_constraint.append(var_u_dict[("u", subset_index, gridpoint_index)])
    mo.addConstr(
        lhs=quicksum(variable for variable in u_constraint), sense=GRB.EQUAL, rhs=1
    )

# Definition of variables chi
# Equation B19
for subset_index in range(detection_total):
    chi_constraint_with_b = []
    chi_constraint = []
    x_constraint = []
    for breakpoint_index in range(1, num_breakpoints):
        chi_constraint_with_b.append(
            var_chi_dict[("chi", subset_index, breakpoint_index)]
            * (exp(b_list[breakpoint_index]) - exp(b_list[breakpoint_index - 1]))
            / 10**10
        )  # Divide by 10**10 to scale the coefficient so that the matrix range is not so large across all the constraints
    for catalog_source_pair in candidate_list:
        x_constraint.append(
            var_x_dict[("x", subset_index, catalog_source_pair)]
            * kappa_rad_dict[catalog_source_pair]
            / 10**10
        )  # Divide by 10**10 to scale the coefficient so that the matrix range is not so large across all the constraints
    mo.addConstr(
        lhs=exp(b_list[0]) * var_chi_dict[("chi", subset_index, 0)]
        + quicksum(variable for variable in chi_constraint_with_b),
        sense=GRB.GREATER_EQUAL,
        rhs=quicksum(variable for variable in x_constraint),
    )

    for breakpoint_index in range(num_breakpoints):
        chi_constraint.append(var_chi_dict[("chi", subset_index, breakpoint_index)])
    for chi_index in range(len(chi_constraint) - 1):
        mo.addConstr(chi_constraint[chi_index] >= chi_constraint[chi_index + 1])

# Definition of variables t
# Equation B20
for subset_index in range(detection_total):
    for gridpoint_index in range(1, len(c_list)):
        mo.addConstr(
            lhs=t_list[subset_index],
            sense=GRB.GREATER_EQUAL,
            rhs=-M * (1 - var_u_dict[("u", subset_index, gridpoint_index)])
            + quicksum(
                1
                / (4 * c_list[gridpoint_index])
                * kappa_dict[product_of_catalog_source_pairs[0]]
                * kappa_dict[product_of_catalog_source_pairs[1]]
                * var_y_dict[
                    (
                        "y",
                        subset_index,
                        product_of_catalog_source_pairs[0],
                        product_of_catalog_source_pairs[1],
                    )
                ]
                * get_distance_2(product_of_catalog_source_pairs, temp_dict)
                for product_of_catalog_source_pairs in list(
                    itertools.combinations(candidate_list, r=2)
                )
                if product_of_catalog_source_pairs[0][1]
                != product_of_catalog_source_pairs[1][1]
            ),
        )

# Definition of variables z
# Equation B21
for subset_index in range(detection_total):
    x_constraint = []
    for catalog_source_pair in candidate_list:
        x_constraint.append(var_x_dict[("x", subset_index, catalog_source_pair)])
    for catalog_index in range(num_catalog + 1):
        mo.addConstr(
            lhs=quicksum(variable for variable in x_constraint),
            sense=GRB.LESS_EQUAL,
            rhs=catalog_index * var_z_dict[("z", subset_index, catalog_index)]
            + num_catalog * (1 - var_z_dict[("z", subset_index, catalog_index)]),
        )
        mo.addConstr(
            lhs=quicksum(variable for variable in x_constraint),
            sense=GRB.GREATER_EQUAL,
            rhs=catalog_index * var_z_dict[("z", subset_index, catalog_index)],
        )

# Definition of variables u
# Equation B222
for subset_index in range(detection_total):
    x_constraint = []
    for catalog_source_pair in candidate_list:
        x_constraint.append(
            var_x_dict[("x", subset_index, catalog_source_pair)]
            * round(kappa_dict[catalog_source_pair], rounding_index)
        )
    for gridpoint_index in range(len(c_list)):
        mo.addConstr(
            lhs=quicksum(variable for variable in x_constraint),
            sense=GRB.LESS_EQUAL,
            rhs=c_list[gridpoint_index]
            * var_u_dict[("u", subset_index, gridpoint_index)]
            + M_2 * (1 - var_u_dict[("u", subset_index, gridpoint_index)]),
        )
        mo.addConstr(
            lhs=quicksum(variable for variable in x_constraint),
            sense=GRB.GREATER_EQUAL,
            rhs=c_list[gridpoint_index]
            * var_u_dict[("u", subset_index, gridpoint_index)],
        )

# Definition of variables p
# Equation B23
for subset_index in range(detection_total):
    x_constraint = []
    for catalog_source_pair in candidate_list:
        x_constraint.append(var_x_dict[("x", subset_index, catalog_source_pair)])
    mo.addConstr(
        lhs=p_list[subset_index],
        sense=GRB.GREATER_EQUAL,
        rhs=ln(2)
        * (
            1
            - quicksum(variable for variable in x_constraint)
            - ln(2) * var_z_dict[("z", subset_index, 0)]
        ),
    )

# Heuristic constraints:
# When 2 sources are more than 5*sigma_max away, they should not be from the same object. Hence, set the y variable corresponding to these 2 sources to 0.
distance_dict = compute_distance_dictionary_2(candidate_list, temp_dict)
dist_constraint_var = []
for pair, distance in distance_dict.items():
    for j in range(detection_total):
        if distance > 5 * sigma_max:
            dist_constraint_var.append(var_y_dict[tuple(["y"] + [j] + list(pair))])
for variable in dist_constraint_var:
    variable.ub = 0

print(f"Set Up Time: {(time.perf_counter() - t1)}")

# Open log file
mo.update()

logfile = open("DirILP_" + str(num_catalog) + "_catalogs.log", "w")
logfile.write("Theoretical optimal solution: %s" % (Bayes_factor(candidate_list, df)))

# Pass data into my callback function
mo._logfile = logfile
mo._vars = mo.getVars()

# Solve model
# mo.Params.MIPGap = 0.01
mo.optimize(mycallback)

# Close log file

logfile.close()
