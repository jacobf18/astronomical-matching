from math import log
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
from astronomical_matching.utils import neg_log_bayes
from astronomical_matching.miqcp import miqcp, setup_miqcp_model
from astronomical_matching.cop_kmeans import run_cop_kmeans
import pandas as pd
import time
import itertools
from typing import Union
from func_timeout import func_timeout, FunctionTimedOut
import os

plt.rcParams["text.usetex"] = True


def simulate_two_objects(
    sigma1: float = 0.04,
    sigma2: float = 0.04,
    distance: float = 0.04,
    num: int = 10,
    seed: int = 0,
):
    """Simulate two overlapping objects.

    Args:
        sigma1 (float, optional): Sigma to use for first object.
            Defaults to 0.04.
        sigma2 (float, optional): Sigma to use for second object.
            Defaults to 0.04.
        distance (float, optional): distance between center of objects.
            Defaults to 0.04.
        num (int, optional): number of sources to generate for each object.
            Defaults to 10.
    """
    np.random.seed(seed)
    center1 = np.array([-distance / 2, 0])
    center2 = np.array([distance / 2, 0])

    sources1 = np.random.multivariate_normal(
        center1, np.eye(2) * (sigma1**2), (num)
    )
    sources2 = np.random.multivariate_normal(
        center2, np.eye(2) * (sigma2**2), (num)
    )

    imageIDs = [0] * (2 * num)
    for i in range(num):
        imageIDs[i] = i
        imageIDs[i + num] = i

    sigmas = ([sigma1] * num) + ([sigma2] * num)
    coords = np.vstack((sources1, sources2))
    df_dict = {
        "ImageID": imageIDs,
        "Sigma": sigmas,
        "coord1 (arcseconds)": coords.T[0],
        "coord2 (arcseconds)": coords.T[1],
    }
    df = pd.DataFrame(df_dict)
    df["kappa"] = df["Sigma"] ** (-2)
    df["SourceID"] = df.index
    df["ObjectID"] = ([0] * num) + ([1] * num)

    return df


def time_method(df: pd.DataFrame, method, repeat: int = 1):
    """Time a method.

    Args:
        df (pd.DataFrame): dataframe with coordinates and uncertainties
        method (function): method to run
        repeat (int, optional): number of time repeats. Defaults to 5.

    Returns:
        list: list of runtimes
    """
    times = []

    for _ in range(repeat):
        time_spent = 0
        try:
            start_time = time.perf_counter()

            func_timeout(5400, method, args=(df,))
            end_time = time.perf_counter()
            time_spent = end_time - start_time
        except FunctionTimedOut:
            time_spent = 1000000000

        times.append(time_spent)
    return times


def log_runtimes(
    filename: str,
    distances: Union[list[float], float],
    num_cats: Union[list[float], float],
    sigma1s: Union[list[float], float],
    sigma2s: Union[list[float], float],
    repeats: int,
):
    if isinstance(distances, float):
        distances = [distances]
    if isinstance(num_cats, int):
        num_cats = [num_cats]
    if isinstance(sigma1s, float):
        sigma1s = [sigma1s]
    if isinstance(sigma2s, float):
        sigma2s = [sigma2s]
    for d, n, s1, s2 in itertools.product(
        distances, num_cats, sigma1s, sigma2s
    ):
        print(d, n, s1, s2)
        for seed in range(repeats):
            pd_dict = {
                "distance": [],
                "number of catalogs": [],
                "sigma 1": [],
                "sigma 2": [],
                "miqcp runtime": [],
                "dirilp runtime": [],
            }
            pd_dict["distance"].append(d)
            pd_dict["number of catalogs"].append(n)
            pd_dict["sigma 1"].append(s1)
            pd_dict["sigma 2"].append(s2)

            df = simulate_two_objects(
                sigma1=s1, sigma2=s2, distance=d, num=n, seed=seed
            )
            if seed == 0:  # check license
                _ = setup_miqcp_model(df, 2, 0, False)
            miqcp_time = time_method(df, miqcp, repeat=1)
            # dirilp_time = time_method(df, setup_dirilp, repeat = 1)
            dirilp_time = [1000000000]
            pd_dict["miqcp runtime"].append(miqcp_time[0])
            pd_dict["dirilp runtime"].append(dirilp_time[0])

            df_times = pd.DataFrame(pd_dict)
            df_times.to_csv(
                filename, mode="a", header=not os.path.exists(filename)
            )


if __name__ == "__main__":
    log_runtimes(
        "runtime-miqcp-sterling.csv", 0.13, [10, 20, 30], 0.04, 0.04, 1
    )
