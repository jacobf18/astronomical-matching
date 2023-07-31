import numpy as np
import pandas as pd


def simulate_objects_on_circle(sigma: float = 0.04,
                               radius: float = 0.1,
                               num_objects: int = 2,
                               num_sources: int = 10,
                               seed: int = 0):
    """Simulate objects on a circle."""

    np.random.seed(seed)

    # Generate equidistant points on a circle
    angles = np.linspace(0, 2 * np.pi, num_objects + 1)[:-1]
    coords = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T

    # Generate sources for each object
    sources = []
    imageIDs = [0] * (num_objects * num_sources)
    objectIDs = [0] * (num_objects * num_sources)
    for i in range(num_objects):
        sources.append(np.random.multivariate_normal(coords[i],
                                                     np.eye(2) * (sigma**2),
                                                     (num_sources)))
        objectIDs[i * num_sources:(i + 1) * num_sources] = [i] * num_sources
        imageIDs[i * num_sources:(i + 1) * num_sources] = list(range(num_sources))

    coords = np.vstack(sources)

    sigmas = [sigma] * (num_objects * num_sources)

    df_dict = {"ImageID": imageIDs,
               "Sigma": sigmas,
                "coord1 (arcseconds)": coords.T[0],
                "coord2 (arcseconds)": coords.T[1]}
    df = pd.DataFrame(df_dict)
    df["kappa"] = df["Sigma"] ** (-2)
    df["SourceID"] = df.index
    df["ObjectID"] = objectIDs

    return df


def simulate_two_objects(sigma1: float = 0.04,
                         sigma2: float = 0.04,
                         distance: float = 0.04,
                         num: int = 10,
                         seed: int = 0):
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

    sources1 = np.random.multivariate_normal(center1,
                                             np.eye(2) * (sigma1**2),
                                             (num))
    sources2 = np.random.multivariate_normal(center2,
                                             np.eye(2) * (sigma2**2),
                                             (num))

    imageIDs = [0] * (2 * num)
    # imageIDs = [0] * (1 * num)
    for i in range(num):
        imageIDs[i] = i
        imageIDs[i+num] = i  # comment out if 1 source

    sigmas = ([sigma1] * num) + ([sigma2] * num)
    coords = np.vstack((sources1, sources2))  # coords = sources1

    df_dict = {"ImageID": imageIDs,
               "Sigma": sigmas,
               "coord1 (arcseconds)": coords.T[0],
               "coord2 (arcseconds)": coords.T[1]}
    df = pd.DataFrame(df_dict)
    df["kappa"] = df["Sigma"] ** (-2)
    df["SourceID"] = df.index
    df["ObjectID"] = ([0] * num) + ([1] * num)

    return df
