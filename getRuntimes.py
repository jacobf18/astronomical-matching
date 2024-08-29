from math import log
import numpy as np
import gurobipy as gp
from astronomical_matching.utils import neg_log_bayes, neg_log_bayes_adjusted
from astronomical_matching.miqcp import miqcp, setup_miqcp_model
from astronomical_matching.cop_kmeans import run_cop_kmeans
from astronomical_matching.kmeans import run_kmeans
from astronomical_matching.chainbreaker import chain_breaking
from astronomical_matching.simulate import simulate_two_objects, simulate_objects_on_circle
import pandas as pd
import time
import itertools
from typing import Union
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import os

MAX_K = 10

def time_method(df: pd.DataFrame, method, repeat: int = 1, has_max_clusters: bool = False):
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
            if has_max_clusters:
                func_timeout(600, method, args=(df,),kwargs={'max_k': MAX_K})
            else:
                func_timeout(600, method, args=(df,))
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
        for seed in tqdm(range(repeats)):
            pd_dict = {
                "distance": [],
                "number of catalogs": [],
                "sigma 1": [],
                "sigma 2": [],
                "miqcp runtime": [],
                "chainbreaker runtime": [],
                "copkmeans runtime": [],
                "kmeans runtime": []
            }
            pd_dict["distance"].append(d)
            pd_dict["number of catalogs"].append(n)
            pd_dict["sigma 1"].append(s1)
            pd_dict["sigma 2"].append(s2)

            df = simulate_two_objects(
                sigma1=s1, sigma2=s2, distance=d, num=n, seed=seed
            )
            # if seed == 0:  # check license
            #     _ = setup_miqcp_model(df, 2, 0, False)
            # print('running miqcp')
            # pd_dict["miqcp runtime"].append(time_method(df, miqcp, repeat=1)[0])
            pd_dict['miqcp runtime'].append(1000000000)
            # print('running chainbreaker')
            # pd_dict['chainbreaker runtime'].append(time_method(df, chain_breaking, repeat=1, has_max_clusters=False)[0])
            pd_dict['chainbreaker runtime'].append(1000000000)
            # print('running copkmeans')
            pd_dict['copkmeans runtime'].append(time_method(df, run_cop_kmeans, repeat=1, has_max_clusters=True)[0])
            # print('running kmeans')
            pd_dict['kmeans runtime'].append(time_method(df, run_kmeans, repeat=1, has_max_clusters=True)[0])

            df_times = pd.DataFrame(pd_dict)
            df_times.to_csv(
                filename, mode="a", header=not os.path.exists(filename)
            )


if __name__ == "__main__":
    log_runtimes(
        filename=f"runtime-all-max_k_{MAX_K}.csv", 
        distances=0.13, 
        num_cats=[150, 200, 300, 400, 500, 1000,5000,10000], 
        sigma1s=0.04, 
        sigma2s=0.04, 
        repeats=20
    )
