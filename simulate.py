import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


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


def main():
    # Parameters
    np.random.seed(42)
    num = 100
    size = 30
    num_catalog = 5

    Qmat = np.empty((1, 2))
    m = mock(num=num, size=size, seed=20)
    print(m)

    # generate catalogs with sigma=0.1" and different selection intevals
    cmat = []
    sel_list = []

    for _ in range(num_catalog):  # List of catalogs with coordinates and conditions
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

    # fig = plt.figure(figsize=(20, 20))
    # for i in range(num_catalog):
    #     x, y = cmat[i][:, :2].T
    #     plt.scatter(x, y)
    # plt.show()


if __name__ == "__main__":
    main()
