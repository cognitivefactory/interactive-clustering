# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.kmeans
* Description:  Implementation of constrained kmeans clustering algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Dependency needed to shuffle data and set random seed.
import random

# Python code typing (mypy).
from typing import Dict, List, Optional, Union

# Dependencies needed to handle float and matrix.
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

# Dependency needed to compute distance between two data points.
from sklearn.metrics import pairwise_distances

# The needed clustering abstract class and utilities methods.
from cognitivefactory.interactive_clustering.clustering.abstract import (
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)

# Dependency needed to manage constraints.
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager
from cognitivefactory.interactive_clustering.utils import checking


# ==============================================================================
# KMEANS CONSTRAINED CLUSTERING
# ==============================================================================
class KMeansConstrainedClustering(AbstractConstrainedClustering):
    """
    This class implements the kmeans constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    References:
        - KMeans Clustering: `MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the fifth Berkeley symposium on mathematical statistics and probability 1(14), 281–297.`
        - Constrained _'COP'_ KMeans Clustering: `Wagstaff, K., C. Cardie, S. Rogers, et S. Schroedl (2001). Constrained K-means Clustering with Background Knowledge. International Conference on Machine Learning`
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(
        self,
        model: str = "COP",
        max_iteration: int = 150,
        tolerance: float = 1e-4,
        random_seed: Optional[int] = None,
        **kargs,
    ) -> None:
        """
        The constructor for KMeans Constrainted Clustering class.

        Args:
            model (str, optional): The kmeans clustering model to use. Available kmeans models are `"COP"`. Defaults to `"COP"`.
            max_iteration (int, optional): The maximum number of kmeans iteration for convergence. Defaults to `150`.
            tolerance (float, optional): The tolerance for convergence computation. Defaults to `1e-4`.
            random_seed (Optional[int]): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Store `self.`model`.
        if model != "COP":  # TODO use `not in {"COP"}`.
            raise ValueError("The `model` '" + str(model) + "' is not implemented.")
        self.model: str = model

        # Store 'self.max_iteration`.
        if max_iteration < 1:
            raise ValueError("The `max_iteration` must be greater than or equal to 1.")
        self.max_iteration: int = max_iteration

        # Store `self.tolerance`.
        if tolerance < 0:
            raise ValueError("The `tolerance` must be greater than 0.0.")
        self.tolerance: float = tolerance

        # Store `self.random_seed`.
        self.random_seed: Optional[int] = random_seed

        # Store `self.kargs` for kmeans clustering.
        self.kargs = kargs

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters: Optional[Dict[str, int]] = None

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================
    def cluster(
        self,
        vectors: Dict[str, Union[ndarray, csr_matrix]],
        nb_clusters: int,
        constraints_manager: Optional[AbstractConstraintsManager] = None,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the KMeans model.

        Args:
            vectors (Dict[str,Union[ndarray,csr_matrix]]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager` (if `constraints_manager` is set). The value of the dictionary represent the vector of each data. Vectors can be dense (`numpy.ndarray`) or sparse (`scipy.sparse.csr_matrix`).
            nb_clusters (int): The number of clusters to compute. #TODO Set defaults to None with elbow method or other method ?
            constraints_manager (Optional[AbstractConstraintsManager], optional): A constraints manager over data IDs that will force clustering to respect some conditions during computation. The list of data IDs managed by `constraints_manager` has to refer to `vectors` keys. If `None`, no constraint are used during the clustering. Defaults to `None`.
            verbose (bool, optional): Enable verbose output. Defaults to `False`.
            **kargs (dict): Other parameters that can be used in the clustering.

        Raises:
            ValueError: if `vectors` and `constraints_manager` are incompatible, or if some parameters are incorrectly set.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """

        ###
        ### GET PARAMETERS
        ###

        # Get `list_of_data_IDs`.
        if not isinstance(vectors, dict):
            raise ValueError("The `vectors` parameter has to be a `dict` type.")
        self.list_of_data_IDs: List[str] = sorted(vectors.keys())
        if not isinstance(self.list_of_data_IDs, list) or not all(
            isinstance(element, str) for element in self.list_of_data_IDs
        ):
            raise ValueError("The `list_of_data_IDs` variable has to be a `list` type.")

        # Check `constraints_manager`.
        self.constraints_manager: AbstractConstraintsManager = checking.check_constraints_manager(
            list_of_data_IDs=self.list_of_data_IDs,
            constraints_manager=constraints_manager,
        )

        # Check `vectors`.
        self.vectors: Dict[str, Union[ndarray, csr_matrix]] = checking.check_vectors(
            list_of_data_IDs=self.list_of_data_IDs,
            vectors=vectors,
        )

        # Store `self.nb_clusters`.
        if nb_clusters < 2:
            raise ValueError("The `nb_clusters` '" + str(nb_clusters) + "' must be greater than or equal to 2.")
        self.nb_clusters: int = nb_clusters

        ###
        ### RUN KMEANS CONSTRAINED CLUSTERING
        ###

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters = None

        # Case of `"COP"` KMeans clustering.
        ##### DEFAULTS : if self.model=="COP":
        self.dict_of_predicted_clusters = self._clustering_kmeans_model_COP(verbose=verbose)

        ###
        ### RETURN PREDICTED CLUSTERS
        ###

        return self.dict_of_predicted_clusters

    # ==============================================================================
    # IMPLEMENTATION - COP KMEANS CLUSTERING
    # ==============================================================================
    def _clustering_kmeans_model_COP(self, verbose: bool = False) -> Dict[str, int]:
        """
        Implementation of COP-kmeans algorithm, based on constraints violation check during cluster affectation.

        References:
            - Constrained _'COP'_ KMeans Clustering: `Wagstaff, K., C. Cardie, S. Rogers, et S. Schroedl (2001). Constrained K-means Clustering with Background Knowledge. International Conference on Machine Learning`

        Args:
            verbose (bool, optional): Enable verbose output. Defaults is `False`.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """

        ###
        ### INITIALIZATION OF CENTROIDS
        ###

        # Initialize `self.centroids`.
        self.centroids: Dict[int, Union[ndarray, csr_matrix]] = self.initialize_centroids()

        # Initialize clusters
        self.clusters: Dict[str, int] = {data_ID: -1 for data_ID in self.list_of_data_IDs}

        ###
        ### RUN KMEANS UNTIL CONVERGENCE
        ###

        # Define variable of convergence checks.
        converged: bool = False
        shift: float = float("Inf")
        current_iteration: int = 0

        # While no convergence
        while (not converged) and (current_iteration < self.max_iteration):

            # Increase current iteration number.
            current_iteration += 1

            ###
            ### ASSIGN DATA TO THE CLOSEST CLUSTER WITH CONSTRAINTS RESPECT
            ###

            # Initialize temporary results.
            new_clusters: Dict[str, int] = {data_ID: -1 for data_ID in self.list_of_data_IDs}

            # For each data ID.
            list_of_data_IDs_to_assign: List[str] = self.list_of_data_IDs.copy()

            # While all data aren't assigned.
            while list_of_data_IDs_to_assign:

                # Get a data_ID to assign
                data_ID_to_assign: str = list_of_data_IDs_to_assign.pop()

                # Get the list of not compatible cluster IDs for assignation
                not_compatible_cluster_IDs: List[int] = [
                    new_clusters[data_ID]
                    for data_ID in self.list_of_data_IDs
                    if (new_clusters[data_ID] != -1)
                    and (
                        self.constraints_manager.get_inferred_constraint(
                            data_ID1=data_ID_to_assign,
                            data_ID2=data_ID,
                        )
                        == "CANNOT_LINK"
                    )
                ]

                # Get the list of possible cluster IDs for assignation.
                possible_cluster_IDs: List[int] = [
                    cluster_ID for cluster_ID in self.centroids.keys() if cluster_ID not in not_compatible_cluster_IDs
                ]

                # If there is possible clusters...
                if possible_cluster_IDs:

                    # Compute distance between data ID and all possible centroids.
                    distances_to_cluster_ID: Dict[float, int] = {
                        pairwise_distances(
                            X=self.vectors[data_ID_to_assign],
                            Y=self.centroids[cluster_ID],
                            metric="euclidean",  # TODO get different pairwise_distances config in **kargs
                        )[0][0].astype(np.float64): cluster_ID
                        for cluster_ID in possible_cluster_IDs
                    }

                    # Get the clostest cluster to data ID by distance.
                    min_distance: float = min(distances_to_cluster_ID)
                    new_clusters[data_ID_to_assign] = distances_to_cluster_ID[min_distance]

                # If there is not possible cluster...
                else:

                    # Assign the data ID to a new cluster
                    new_clusters[data_ID_to_assign] = max(
                        max([cluster_ID for _, cluster_ID in new_clusters.items()]) + 1, self.nb_clusters
                    )

                # Assign all data ID that are linked by a `"MUST_LINK"` constraint to the new cluster.
                for data_ID in self.list_of_data_IDs:
                    if (
                        self.constraints_manager.get_inferred_constraint(
                            data_ID1=data_ID_to_assign,
                            data_ID2=data_ID,
                        )
                        == "MUST_LINK"
                    ):
                        if data_ID in list_of_data_IDs_to_assign:
                            list_of_data_IDs_to_assign.remove(data_ID)
                        new_clusters[data_ID] = new_clusters[data_ID_to_assign]

            # Rename cluster IDs by order.
            new_clusters = rename_clusters_by_order(clusters=new_clusters)

            ###
            ### COMPUTE NEW CENTROIDS
            ###

            # Compute new centroids
            new_centroids: Dict[int, Union[ndarray, csr_matrix]] = self.compute_centroids(clusters=new_clusters)

            ###
            ### CHECK CONVERGENCE
            ###

            # Check by centroids difference (with tolerance).
            if set(self.clusters.values()) == set(new_clusters.values()):

                # Compute shift between kmeans iterations.
                shift = sum(
                    pairwise_distances(X=self.centroids[cluster_ID], Y=new_centroids[cluster_ID], metric="euclidean")[
                        0
                    ][0]
                    for cluster_ID in self.centroids
                )

                # If shift is under tolerance : convergence !
                converged = shift < self.tolerance

            # Check if number of clusters have changed.
            else:

                # Uncomparable shift.
                shift = float("Inf")

                # Don't converge.
                converged = False

            ###
            ### APPLY NEW COMPUTATIONS
            ###

            # Affect new clusters.
            self.clusters = new_clusters.copy()

            # Affect new centroids.
            self.centroids = new_centroids.copy()

            # Verbose.
            if verbose and (current_iteration % 5 == 0):
                # Verbose - Print progression status.
                print("    CLUSTERING_ITERATION=" + str(current_iteration), ",", "shift=" + str(shift))

        # Verbose.
        if verbose:
            # Verbose - Print progression status.
            print("    CLUSTERING_ITERATION=" + str(current_iteration), ",", "converged=" + str(converged))

        return self.clusters

    # ==============================================================================
    # INITIALIZATION OF CLUSTERS
    # ==============================================================================
    def initialize_centroids(
        self,
    ) -> Dict[int, Union[ndarray, csr_matrix]]:
        """
        Initialize the centroid of each cluster by a vector.
        The choice is based on a random selection among data to cluster.

        Returns:
            Dict[int, Union[ndarray,csr_matrix]]: A dictionary which represent each cluster by a centroid.
        """

        # Get the list of possible indices.
        indices: List[str] = self.list_of_data_IDs.copy()

        # Shuffle the list of possible indices.
        random.seed(self.random_seed)
        random.shuffle(indices)

        # Subset indices.
        indices = indices[: self.nb_clusters]

        # Set initial centroids based on vectors.
        centroids: Dict[int, Union[ndarray, csr_matrix]] = {
            cluster_ID: self.vectors[indices[cluster_ID]] for cluster_ID in range(self.nb_clusters)
        }

        # Return centroids.
        return centroids

    # ==============================================================================
    # COMPUTE NEW CENTROIDS
    # ==============================================================================
    def compute_centroids(
        self,
        clusters: Dict[str, int],
    ) -> Dict[int, Union[ndarray, csr_matrix]]:
        """
        Compute the centroids of each cluster.

        Args:
            clusters (Dict[str,int]): Current clusters assignation.

        Returns:
            Dict[int, Union[ndarray,csr_matrix]]: A dictionary which represent each cluster by a centroid.
        """

        # Initialize centroids.
        centroids: Dict[int, Union[ndarray, csr_matrix]] = {}

        # For all possible cluster ID.
        for cluster_ID in set(clusters.values()):

            # Compute cluster members.
            members_of_cluster_ID = [
                vector for data_ID, vector in self.vectors.items() if clusters[data_ID] == cluster_ID
            ]

            # Compute centroid.
            centroid_for_cluster_ID: Union[ndarray, csr_matrix] = sum(members_of_cluster_ID) / len(
                members_of_cluster_ID
            )

            # Add centroids.
            centroids[cluster_ID] = centroid_for_cluster_ID

        # Return centroids.
        return centroids
