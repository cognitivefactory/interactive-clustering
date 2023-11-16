# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.mpckmeans
* Description:  Implementation of constrained mpckmeans clustering algorithms.
* Author:       Esther LENOTRE, David NICOLAZO, Marc TRUTT
* Created:      10/09/2022
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import warnings

# import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.clustering.abstract import (
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager


# ==============================================================================
# MPCKMEANS CONSTRAINED CLUSTERING
# ==============================================================================
class MPCKMeansConstrainedClustering(AbstractConstrainedClustering):
    """
    This class implements the MPCkmeans constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    Forked from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
    Modified by Esther LENOTRE <git@estherlenotre.fr> according to https://proceedings.mlr.press/v5/givoni09a.html

    References:
        - KMeans Clustering: `MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the fifth Berkeley symposium on mathematical statistics and probability 1(14), 281–297.`
        - Constrained _'MPC'_ KMeans Clustering: `Khan, Md. A., Tamim, I., Ahmed, E., & Awal, M. A. (2012). Multiple Parameter Based Clustering (MPC): Prospective Analysis for Effective Clustering in Wireless Sensor Network (WSN) Using K-Means Algorithm. In Wireless Sensor Network (Vol. 04, Issue 01, pp. 18–24). Scientific Research Publishing, Inc. https://doi.org/10.4236/wsn.2012.41003`

    Example:
        ```python
        # Import.
        from scipy.sparse import csr_matrix
        from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
        from cognitivefactory.interactive_clustering.clustering.dbscan import MPCKMeansConstrainedClustering

        # NB : use cognitivefactory.interactive_clustering.utils to preprocess and vectorize texts.

        vectors = {
            "0": csr_matrix([1.00, 0.00, 0.00, 0.00]),
            "1": csr_matrix([0.95, 0.02, 0.02, 0.01]),
            "2": csr_matrix([0.98, 0.00, 0.02, 0.00]),
            "3": csr_matrix([0.99, 0.00, 0.01, 0.00]),
            "4": csr_matrix([0.50, 0.22, 0.21, 0.07]),
            "5": csr_matrix([0.50, 0.21, 0.22, 0.07]),
            "6": csr_matrix([0.01, 0.01, 0.01, 0.97]),
            "7": csr_matrix([0.00, 0.01, 0.00, 0.99]),
            "8": csr_matrix([0.00, 0.00, 0.00, 1.00]),
        }

        # Define constraints manager.
        constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))
        constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="0", data_ID2="7", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="0", data_ID2="8", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="4", data_ID2="5", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")
        constraints_manager.add_constraint(data_ID1="2", data_ID2="4", constraint_type="CANNOT_LINK")

        cluster_model = MPCKMeansConstrainedClustering()
        dict_of_predicted_clusters = cluster_model.cluster(
            constraints_manager=constraints_manager,
            vectors=vectors,
            nb_clusters=3,
        )

        # Print results.
        print("Expected results", ";", {"0": 0, "1": 0, "2": 1, "3": 1, "4": 2, "5": 2, "6": 0, "7": 0, "8": 0,})
        print("Computed results", ":", dict_of_predicted_clusters)
        ```

    Warns:
        FutureWarning: `clustering.mpckmeans.MPCKMeansConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(
        self,
        model: str = "MPC",
        max_iteration: int = 150,
        w: float = 1.0,
        random_seed: Optional[int] = None,
        **kargs,
    ) -> None:
        """
        The constructor for MPCKMeans Constrainted Clustering class.

        Args:
            model (str, optional): The kmeans clustering model to use. Available kmeans models are `"MPC"`. Defaults to `"MPC"`.
            max_iteration (int, optional): The maximum number of kmeans iteration for convergence. Defaults to `150`.
            w (float, optional): Weight for the constraints
            random_seed (Optional[int]): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Warns:
            FutureWarning: `clustering.mpckmeans.MPCKMeansConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Deprecation warnings
        warnings.warn(
            "`clustering.mpckmeans.MPCKMeansConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.",
            FutureWarning,  # DeprecationWarning
            stacklevel=2,
        )

        # Store `self.`model`.
        if model != "MPC":  # TODO use `not in {"MPC"}`.
            raise ValueError("The `model` '" + str(model) + "' is not implemented.")
        self.model: str = model

        # Store 'self.max_iteration`.
        if max_iteration < 1:
            raise ValueError("The `max_iteration` must be greater than or equal to 1.")
        self.max_iteration: int = max_iteration

        # Store `self.weight`.
        if w < 0:
            raise ValueError("The `weight` must be greater than 0.0.")
        self.w: float = w

        # Store `self.random_seed`.
        self.random_seed: Optional[int] = random_seed

        # Store `self.kargs` for clustering.
        self.kargs = kargs

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters: Optional[Dict[str, int]] = None

        # Initialize `ml_graph` and `cl_graph`.
        self.ml_graph: Dict[str, List[str]] = {}
        self.cl_graph: Dict[str, List[str]] = {}

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================

    def cluster(
        self,
        constraints_manager: AbstractConstraintsManager,
        vectors: Dict[str, csr_matrix],
        nb_clusters: Optional[int],
        verbose: bool = False,
        y=None,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the KMeans model.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs that will force clustering to respect some conditions during computation.
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data.
            nb_clusters (Optional[int]): The number of clusters to compute. Here None.
            verbose (bool, optional): Enable verbose output. Defaults to `False`.
            y : Something.
            **kargs (dict): Other parameters that can be used in the clustering.

        Raises:
            ValueError: if `vectors` and `constraints_manager` are incompatible, or if some parameters are incorrectly set.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """

        ###
        ### GET PARAMETERS
        ###

        # Store `self.constraints_manager` and `self.list_of_data_IDs`.
        if not isinstance(constraints_manager, AbstractConstraintsManager):
            raise ValueError("The `constraints_manager` parameter has to be a `AbstractConstraintsManager` type.")
        self.constraints_manager: AbstractConstraintsManager = constraints_manager
        self.list_of_data_IDs: List[str] = self.constraints_manager.get_list_of_managed_data_IDs()

        # Store `self.vectors`.
        if not isinstance(vectors, dict):
            raise ValueError("The `vectors` parameter has to be a `dict` type.")
        self.vectors: Dict[str, csr_matrix] = vectors

        # Store `self.nb_clusters`.
        if (nb_clusters is None) or (nb_clusters < 2):
            raise ValueError("The `nb_clusters` '" + str(nb_clusters) + "' must be greater than or equal to 2.")
        self.nb_clusters: int = min(nb_clusters, len(self.list_of_data_IDs))

        # TODO: Reformat vectors
        id_names: np.ndarray = np.array(list(vectors.keys()))
        X: np.ndarray = np.array(
            [
                (np.array(v).flatten() if isinstance(v, (np.ndarray, list)) else v.toarray().flatten())
                for v in vectors.values()
            ]
        )

        # TODO: reformat constraints
        self.ml: List[Tuple[int, int]] = [
            (i, j)
            for i, data_ID_i in enumerate(self.list_of_data_IDs)
            for j, data_ID_j in enumerate(self.list_of_data_IDs)
            if (
                self.constraints_manager.get_inferred_constraint(
                    data_ID1=data_ID_j,
                    data_ID2=data_ID_i,
                )
                == "MUST_LINK"
            )
        ]

        # TODO: reformat constraints
        self.cl: List[Tuple[int, int]] = [
            (i, j)
            for i, data_ID_i in enumerate(self.list_of_data_IDs)
            for j, data_ID_j in enumerate(self.list_of_data_IDs)
            if (
                self.constraints_manager.get_inferred_constraint(
                    data_ID1=data_ID_j,
                    data_ID2=data_ID_i,
                )
                == "CANNOT_LINK"
            )
        ]

        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = self.preprocess_constraints()

        # Initialize cluster centers
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

        # Initialize metrics
        A = np.identity(X.shape[1])

        iteration = 0
        converged = False

        # Repeat until convergence
        while not converged and iteration < self.max_iteration:
            prev_cluster_centers = cluster_centers.copy()

            # Find farthest pair of points according to each metric
            farthest = self._find_farthest_pairs_of_points(X, A)

            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, A, farthest, ml_graph, cl_graph, self.w)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Update metrics
            A = self._update_metrics(X, labels, cluster_centers, farthest, ml_graph, cl_graph, self.w)

            # Check for convergence
            cluster_centers_shift = prev_cluster_centers - cluster_centers
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)
            iteration += 1

        self.cluster_centers = cluster_centers
        self.labels = labels

        self.dict_of_predicted_clusters = {id_names[i]: self.labels[i] for i, label in enumerate(labels) if label != -1}

        self.dict_of_predicted_clusters = rename_clusters_by_order(clusters=self.dict_of_predicted_clusters)

        for data_ID in self.list_of_data_IDs:
            if data_ID not in self.dict_of_predicted_clusters.keys():
                self.dict_of_predicted_clusters[data_ID] = -1

        return self.dict_of_predicted_clusters

    # ==============================================================================
    # USEFUL FUNCTIONS
    # ==============================================================================

    def dist(self, i: int, S: List[int], points: np.ndarray) -> float:
        """
        Computes the minimum distance of a single point to a group of points.

        Args:
            i (int): Index of the single point.
            S (List[int]): List of the index of the group of points .
            points (np.ndarray): Array containing all the points.

        Returns:
            float: Minimum distance of the single to the group of points.
        """

        distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
        return distances.min()

    def _dist(self, x: np.ndarray, y: np.ndarray, A: np.ndarray) -> float:
        """
        Computes the Mahalanobis distance between two points.
        "(x - y)^T A (x - y)"

        Args:
            x (np.ndarray): First point.
            y (np.ndarray): Second point .
            A (np.ndarray): Inverse of a covariance matrix.

        Returns:
            float: Minimum distance of the single to the group of points.
        """

        return scipy.spatial.distance.mahalanobis(x, y, A) ** 2

    def _find_farthest_pairs_of_points(self, X: np.ndarray, A: np.ndarray) -> Tuple[int, int, float]:
        """
        Finds the farthest pair of points.

        Args:
            X (np.ndarray): Set of points.
            A (np.ndarray): Positive-definite matrix used for the distances.

        Returns:
            Tuple[int, int, float]: Indexes of the farthest pair of points and the corresponding distance.
        """

        farthest = None
        n = X.shape[0]
        max_distance = 0.0

        for i in range(n):
            for j in range(n):
                if j < i:
                    distance = self._dist(X[i], X[j], A)
                    if distance > max_distance:
                        max_distance = distance
                        farthest = (i, j, distance)

        assert farthest is not None

        return farthest

    def weighted_farthest_first_traversal(self, points: np.ndarray, weights: np.ndarray, k: int) -> List[int]:
        """
        Applies weighted farthest first traversal algorithm.

        Args:
            points (np.ndarray): Set of points.
            weights (np.ndarray): Weights for the distances.
            k (int): Number of points to be traversed

        Returns:
            List[int]: Indexes of the traversed points.
        """
        traversed = []

        # Choose the first point randomly (weighted)
        i = np.random.choice(len(points), size=1, p=weights)[0]
        traversed.append(i)

        # Find remaining n - 1 maximally separated points
        for _ in range(k - 1):
            max_dst, max_dst_index = 0, None

            number_of_points = len(points)
            for j in range(number_of_points):
                if j not in traversed:
                    dst = self.dist(j, traversed, points)
                    weighted_dst = weights[j] * dst

                    if weighted_dst > max_dst:
                        max_dst = weighted_dst
                        max_dst_index = j

            traversed.append(max_dst_index)

        return traversed

    # ==============================================================================
    # INITIALIZATION OF CLUSTERS
    # ==============================================================================
    def _add_both(self, d, i, j):
        d[i].add(j)
        d[j].add(i)

    def _dfs(self, i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                self._dfs(j, graph, visited, component)
        component.append(i)

    def preprocess_constraints(
        self,
    ) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], List[List[int]]]:
        """
        Initialize each cluster.
        The choice is based on the neighborhoods created by the initial constraints.

        Raises:
            ValueError: if there is a Cannot-link constraint in conflict with a Must-link constraint involving both one same point.

        Returns:
            Tuple[Dict[int, set], Dict[int, set], List[List[int]]]:
            A new list of must-link and cannot-link constraints as well as the lambda starting neighborhoods.
        """

        # Get the list of possible indices.
        indices: List[str] = self.list_of_data_IDs.copy()

        n: int = len(indices)

        # Represent the graphs using adjacency-lists.
        ml_graph: Dict[int, Set[int]] = {}
        cl_graph: Dict[int, Set[int]] = {}

        for k in range(n):
            ml_graph[k] = set()
            cl_graph[k] = set()

        for data_ID_i1, data_ID_j1 in self.ml:
            ml_graph[data_ID_i1].add(data_ID_j1)
            ml_graph[data_ID_j1].add(data_ID_i1)

        for data_ID_i2, data_ID_j2 in self.cl:
            cl_graph[data_ID_i2].add(data_ID_j2)
            cl_graph[data_ID_j2].add(data_ID_i2)

        visited = [False for _ in range(n)]
        neighborhoods = []
        for index in range(n):
            if not visited[index] and ml_graph[index]:
                component: List[int] = []
                self._dfs(index, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
                neighborhoods.append(component)

        for data_ID_i3, data_ID_j3 in self.cl:
            for x in ml_graph[data_ID_i3]:
                self._add_both(cl_graph, x, data_ID_j3)

            for y in ml_graph[data_ID_j3]:
                self._add_both(cl_graph, data_ID_i3, y)

            for a in ml_graph[data_ID_i3]:
                for b in ml_graph[data_ID_j3]:
                    self._add_both(cl_graph, a, b)

        for index_1, _ in ml_graph.items():
            for index_2 in ml_graph[index_1]:
                if index_2 != index_1 and index_2 in cl_graph[index_1]:
                    raise ValueError("Inconsistent constraints between " + str(index_1) + " and " + str(index_2))

        return ml_graph, cl_graph, neighborhoods

    def _initialize_cluster_centers(self, X: np.ndarray, neighborhoods: List[List[int]]) -> np.ndarray:
        """
        Initialises cluster centers.

        Args:
            X (np.ndarray): Set of points.
            neighborhoods (List[List[int]]): Lists of neighbors for each point.

        Returns:
            np.ndarray: Computed centers.
        """

        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])
        neighborhood_weights = neighborhood_sizes / neighborhood_sizes.sum()

        # print('\t', len(neighborhoods), neighborhood_sizes)

        if len(neighborhoods) > self.nb_clusters:
            cluster_centers = neighborhood_centers[
                self.weighted_farthest_first_traversal(neighborhood_centers, neighborhood_weights, self.nb_clusters)
            ]
        else:
            if neighborhoods:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            if len(neighborhoods) < self.nb_clusters:
                remaining_cluster_centers = X[
                    np.random.choice(X.shape[0], self.nb_clusters - len(neighborhoods), replace=False), :
                ]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

    # ==============================================================================
    # COMPUTE CLUSTERS
    # ==============================================================================

    def _f_m(self, X: np.ndarray, i: int, j: int, A) -> float:
        return self._dist(X[i], X[j], A)

    def _f_c(self, X: np.ndarray, i: int, j: int, A, farthest) -> float:
        return farthest[2] - self._dist(X[i], X[j], A)

    def _objective_fn(
        self, X: np.ndarray, i: int, labels, cluster_centers, cluster_id, A, farthest, ml_graph, cl_graph, w
    ) -> float:
        sign, logdet = np.linalg.slogdet(A)
        log_det_a = sign * logdet

        if log_det_a == np.inf:
            log_det_a = 0

        term_d: float = self._dist(X[i], cluster_centers[cluster_id], A) - log_det_a

        term_m: float = 0
        for j in ml_graph[i]:
            if labels[j] >= 0 and labels[j] != cluster_id:
                term_m += 2 * w * self._f_m(X, i, j, A)

        term_c: float = 0
        for k in cl_graph[i]:
            if labels[k] == cluster_id:
                # assert self._f_c(i, k, A, farthest) >= 0
                term_c += 2 * w * self._f_c(X, i, k, A, farthest)

        return term_d + term_m + term_c

    def _assign_clusters(self, X, y, cluster_centers, A, farthest, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for i in index:
            labels[i] = np.argmin(
                [
                    self._objective_fn(X, i, labels, cluster_centers, cluster_id, A, farthest, ml_graph, cl_graph, w)
                    for cluster_id, cluster_center in enumerate(cluster_centers)
                ]
            )

        # Handle empty clusters
        n_samples_in_cluster = np.bincount(labels, minlength=self.nb_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        for empty_cluster_id in empty_clusters:
            # Get clusters that have at least 2 points and can give one of them to another cluster
            filled_clusters = np.where(n_samples_in_cluster > 1)[0]

            # Get points from filled_clusters, and compute distance to their center
            distances_to_clusters: Dict[str, float] = {}

            for cluster_id in filled_clusters:
                available_cluster_points = np.where(labels == cluster_id)[0]
                for point in available_cluster_points:
                    distances_to_clusters[point] = self._dist(X[point], cluster_centers[cluster_id], A)

            # Fill empty clusters with the farthest points regarding their respective centers
            filling_point: float = max(distances_to_clusters, key=distances_to_clusters.get)
            labels[filling_point] = empty_cluster_id

            n_samples_in_cluster = np.bincount(labels, minlength=10)

        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if empty_clusters.size:
            # print("Empty clusters")
            raise ValueError("Empty Clusters Exception")
        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.nb_clusters)])

    # ==============================================================================
    # UPDATE METRICS
    # ==============================================================================

    def _update_metrics(self, X, labels, cluster_centers, farthest, ml_graph, cl_graph, w):
        N, D = X.shape
        A = np.zeros((D, D))

        for d in range(D):
            term_x = np.sum([(x[d] - cluster_centers[labels[i], d]) ** 2 for i, x in enumerate(X)])

            term_m = 0
            for i in range(N):
                for j in ml_graph[i]:
                    if labels[i] != labels[j]:
                        term_m += 1 / 2 * w * (X[i, d] - X[j, d]) ** 2

            term_c = 0
            for k in range(N):
                for m in cl_graph[k]:
                    if labels[k] == labels[m]:
                        tmp = (X[farthest[0], d] - X[farthest[1], d]) ** 2 - (X[k, d] - X[m, d]) ** 2
                        term_c += w * max(tmp, 0)

            # print('term_x', term_x, 'term_m', term_m, 'term_c', term_c)

            A[d, d] = N / max(term_x + term_m + term_c, 1e-9)

        return A
