# -*- coding: utf-8 -*-
# from optparse import Option
import random
# from re import I
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, List, Tuple
# from scipy.spatial.distance import cdist


def cop_kmeans(
    dataset: ArrayLike,
    k: int,
    ml: Optional[List[Tuple[int, int]]] = None,
    cl: Optional[List[Tuple[int, int]]] = None,
    initialization: str = "kmpp",
    max_iter: int = 300,
    tol: float = 1e-4,
    sample_weights: Optional[ArrayLike] = None,
) -> Tuple[Optional[list], Optional[ArrayLike]]:
    """COP-KMeans algorithm

    Args:
        dataset (ArrayLike): 2D array of data points
        k (int): number of clusters
        ml (Optional[list[tuple[int, int]]], optional): Must-Link list. Defaults to None.
        cl (Optional[list[tuple[int, int]]], optional): Cannot-Link list. Defaults to None.
        initialization (str, optional): Type of initialization. Defaults to "kmpp".
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        tol (float, optional): Tolerance for early stopping. Defaults to 1e-4.
        sample_weights (Optional[ArrayLike], optional): Sample weights for weighted kmeans. Defaults to None.

    Returns:
        tuple[Optional[list[int]], Optional[ArrayLike]]: Clusters, Centers
    """
    if cl is None:
        cl = []
    if ml is None:
        ml = []
    if sample_weights is None:
        # sample_weights = [1] * len(dataset)
        sample_weights = np.ones(len(dataset))

    # Initialize COP information
    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset, sample_weights)
    tol = tolerance(tol, dataset)

    # Initialize clusters
    centers = initialize_centers(dataset, k, initialization, sample_weights)

    # Run COP-KMeans
    for _ in range(max_iter):
        clusters_ = [-1] * len(dataset)  # -1 means unassigned

        # Assign points to clusters
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    return None, None

        # Update centers
        clusters_, centers_ = compute_centers(
            clusters_, dataset, k, ml_info, sample_weights
        )

        # Check for convergence
        # shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        shift = np.sum(np.linalg.norm(centers - centers_, ord = 2, axis=1))
        if shift <= tol:
            break

        # Update clusters
        centers = centers_

    return clusters_, centers_


def l2_distance(point1, point2):
    return sum([(float(i) - float(j)) ** 2 for (i, j) in zip(point1, point2)])
    # return np.sum((point1 - point2) ** 2)


def tolerance(tol: float, dataset: ArrayLike):
    # n = len(dataset)
    # dim = len(dataset[0])
    # averages = [sum(dataset[i][d] for i in range(n)) / float(n) for d in range(dim)]
    # variances = [
    #     sum((dataset[i][d] - averages[d]) ** 2 for i in range(n)) / float(n)
    #     for d in range(dim)
    # ]
    # return tol * sum(variances) / dim
    return np.mean(np.var(dataset, axis=0)) * tol


def closest_clusters(centers, datapoint):
    # distances = [l2_distance(center, datapoint) for center in centers]
    distances = np.linalg.norm(centers - datapoint, ord = 2, axis=1)
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances


def initialize_centers(
    dataset: ArrayLike, k: int, method: str, sample_weights: ArrayLike
):
    if method == "random":
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]

    elif method == "kmpp":
        chances = sample_weights
        dim = len(dataset[0])
        centers = np.zeros((k,dim))

        for i in range(k):
            # chances = [x / sum(chances) for x in chances]
            chances = chances / np.sum(chances)
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            # centers.append(dataset[index])
            centers[i] = dataset[index]

            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]

        return centers


def violate_constraints(
    data_index,
    cluster_index,
    clusters: ArrayLike,
    ml: List[Tuple[int, int]],
    cl: List[Tuple[int, int]],
):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False


def compute_centers(
    clusters: ArrayLike, dataset: ArrayLike, k: int, ml_info, sample_weights: ArrayLike
):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]

    dim = len(dataset[0])
    # centers = [[0.0] * dim for i in range(k)]
    centers = np.zeros((k,dim))

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i] * sample_weights[j]
        counts[c] += sample_weights[j]

    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i] / float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [
            sum(l2_distance(centers[clusters[i]], dataset[i]) for i in group)
            for group in ml_groups
        ]
        group_ids = sorted(
            range(len(ml_groups)),
            key=lambda x: current_scores[x] - ml_scores[x],
            reverse=True,
        )
        for j in range(k - k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid

    return clusters, centers


def get_ml_info(
    ml: List[Tuple[int, int]], dataset: ArrayLike, sample_weights: ArrayLike
):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]:
            continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False

    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]

    for j, group in enumerate(groups):
        for d in range(dim):
            count = 0
            for i in group:
                centroids[j][d] += dataset[i][d] * sample_weights[i]
                count += sample_weights[i]
            centroids[j][d] /= count

    scores = [
        sum(l2_distance(centroids[j], dataset[i]) for i in groups[j])
        for j in range(len(groups))
    ]

    return groups, scores, centroids


def transitive_closure(ml: List[Tuple[int, int]], cl: List[Tuple[int, int]], n: int):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception("inconsistent constraints between %d and %d" % (i, j))

    return ml_graph, cl_graph


import cProfile
if __name__ == "__main__":
    # Run test of cop-kmeans

    dataset = np.random.randint(0, 1000, (1000, 2))
    # ml = [(0, 1), (2, 3), (4, 5), (6, 7)]
    # cl = [(0, 2), (4, 6)]

    with cProfile.Profile() as pr:
        clusters, centers = cop_kmeans(dataset, 20, None, None)

        pr.dump_stats('cop2.prof')