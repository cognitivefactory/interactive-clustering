# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.dbscan
* Description:  Implementation of constrained DBScan clustering algorithms.
* Author:       Marc TRUTT, Esther LENOTRE, David NICOLAZO
* Created:      08/05/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import warnings

# import random
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise_distances

from cognitivefactory.interactive_clustering.clustering.abstract import (
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager


# ==============================================================================
# DBSCAN CONSTRAINED CLUSTERING
# ==============================================================================
class DBScanConstrainedClustering(AbstractConstrainedClustering):
    """
    This class implements the DBScan constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    References:
        - DBScan Clustering: `Ester, Martin & Kröger, Peer & Sander, Joerg & Xu, Xiaowei. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. KDD. 96. 226-231`.
        - Constrained DBScan Clustering: `Ruiz, Carlos & Spiliopoulou, Myra & Menasalvas, Ernestina. (2007). C-DBSCAN: Density-Based Clustering with Constraints. 216-223. 10.1007/978-3-540-72530-5_25.`

    Example:
        ```python
        # Import.
        from scipy.sparse import csr_matrix
        from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
        from cognitivefactory.interactive_clustering.clustering.dbscan import DBScanConstrainedClustering

        # Create an instance of CDBscan clustering.
        clustering_model = DBScanConstrainedClustering(
            eps=0.02,
            min_samples=3,
        )

        # Define vectors.
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

        # Run clustering.
        dict_of_predicted_clusters = clustering_model.cluster(
            constraints_manager=constraints_manager,
            vectors=vectors,
            #### nb_clusters=None,
        )

        # Print results.
        print("Expected results", ";", {"0": 0, "1": 0, "2": 1, "3": 1, "4": 2, "5": 2, "6": 0, "7": 0, "8": 0,})
        print("Computed results", ":", dict_of_predicted_clusters)
        ```

    Warns:
        FutureWarning: `clustering.dbscan.DBScanConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        random_seed: Optional[int] = None,
        **kargs,
    ) -> None:
        """
        The constructor for DBScan Constrainted Clustering class.

        Args:
            eps (float): The maximus radius of a neighborhood around its center. Defaults to `0.5`.
            min_samples (int): The minimum number of points in a neighborhood to consider a center as a core point. Defaults to `5`.
            random_seed (Optional[int]): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Warns:
            FutureWarning: `clustering.dbscan.DBScanConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Deprecation warnings
        warnings.warn(
            "`clustering.dbscan.DBScanConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.",
            FutureWarning,  # DeprecationWarning
            stacklevel=2,
        )

        # Store 'self.eps`.
        if eps <= 0:
            raise ValueError("The `eps` must be greater than 0.")
        self.eps: float = eps

        # Store 'self.min_samples`.
        if min_samples <= 0:
            raise ValueError("The `min_samples` must be greater than or equal to 1.")
        self.min_samples: int = min_samples

        # Store `self.random_seed`.
        self.random_seed: Optional[int] = random_seed

        # Store `self.kargs` for kmeans clustering.
        self.kargs = kargs

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters: Optional[Dict[str, int]] = None

        # Initialize number of clusters attributes.
        self.number_of_single_noise_point_clusters: int = 0
        self.number_of_regular_clusters: int = 0
        self.number_of_clusters: int = 0

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================
    def cluster(
        self,
        constraints_manager: AbstractConstraintsManager,
        vectors: Dict[str, csr_matrix],
        nb_clusters: Optional[int] = None,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the DBScan model.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs that will force clustering to respect some conditions during computation.
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data.
            nb_clusters (Optional[int]): The number of clusters to compute. Here `None`.
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
        if nb_clusters is not None:
            raise ValueError("The `nb_clusters` should be 'None' for DBScan clustering.")
        self.nb_clusters: Optional[int] = None

        ###
        ### COMPUTE DISTANCE
        ###

        # Compute pairwise distances.
        matrix_of_pairwise_distances: csr_matrix = pairwise_distances(
            X=vstack(self.vectors[data_ID] for data_ID in self.constraints_manager.get_list_of_managed_data_IDs()),
            metric="euclidean",  # TODO get different pairwise_distances config in **kargs
        )

        # Format pairwise distances in a dictionary and store `self.dict_of_pairwise_distances`.
        self.dict_of_pairwise_distances: Dict[str, Dict[str, float]] = {
            vector_ID1: {
                vector_ID2: float(matrix_of_pairwise_distances[i1, i2])
                for i2, vector_ID2 in enumerate(self.constraints_manager.get_list_of_managed_data_IDs())
            }
            for i1, vector_ID1 in enumerate(self.constraints_manager.get_list_of_managed_data_IDs())
        }

        ###
        ### INITIALIZE VARIABLES
        ###

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters = {}

        # To assign "CORE", "SINGLE_CORE" or "NOISE" labels to the points
        self.dict_of_data_IDs_labels: Dict[str, str] = {data_ID: "UNLABELED" for data_ID in self.list_of_data_IDs}

        # To store the lists of points of each computed local cluster
        self.dict_of_local_clusters: Dict[str, List[str]] = {}

        # To store the lists of points of each computed core local cluster
        self.dict_of_core_local_clusters: Dict[str, List[str]] = {data_ID: [] for data_ID in self.list_of_data_IDs}

        ###
        ### CREATE LOCAL CLUSTERS
        ###

        for possible_core_ID in self.list_of_data_IDs:
            if self.dict_of_data_IDs_labels[possible_core_ID] != "SINGLE_CORE":
                # Points involved in a Cannot-link constraint are not associated to other points in this step
                list_of_possible_neighbors: List[str] = [
                    neighbor_ID
                    for neighbor_ID in self.list_of_data_IDs
                    if self.dict_of_data_IDs_labels[neighbor_ID] != "SINGLE_CORE"
                ]

                # Compute distances to other possible neighbors
                distances_to_possible_neighbors: Dict[str, float] = {
                    neighbor_ID: self.dict_of_pairwise_distances[possible_core_ID][neighbor_ID]
                    for neighbor_ID in list_of_possible_neighbors
                }

                # Keep only points within the radius of eps as neighbors
                list_of_neighbors_ID: List[str] = [
                    neighbor_ID
                    for neighbor_ID in list_of_possible_neighbors
                    if distances_to_possible_neighbors[neighbor_ID] <= self.eps
                ]

                # Get the lists of not compatible data_IDs for deciding if the points are separated in different clusters
                not_compatible_cluster_IDs: List[List[str]] = [
                    [
                        data_ID_i
                        for data_ID_i in list_of_neighbors_ID
                        if (
                            self.constraints_manager.get_inferred_constraint(
                                data_ID1=data_ID_j,
                                data_ID2=data_ID_i,
                            )
                            == "CANNOT_LINK"
                        )
                    ]
                    for data_ID_j in list_of_neighbors_ID
                ]

                # Check if there is a Cannot-link constraint between points in the neighborhood
                no_conflict = True
                for neighborhood_not_compatible_IDs in not_compatible_cluster_IDs:
                    if neighborhood_not_compatible_IDs:
                        no_conflict = False
                        break

                if len(list_of_neighbors_ID) < self.min_samples:
                    self.dict_of_data_IDs_labels[possible_core_ID] = "NOISE"

                elif no_conflict is False:
                    for neighbor_ID in list_of_neighbors_ID:
                        # Each point of the neighborhood will be a single core point cluster
                        # and won't be involved in other clusters in this step

                        self.dict_of_data_IDs_labels[neighbor_ID] = "SINGLE_CORE"
                        self.dict_of_local_clusters[neighbor_ID] = [neighbor_ID]

                else:
                    self.dict_of_data_IDs_labels[possible_core_ID] = "CORE"
                    self.dict_of_local_clusters[possible_core_ID] = list_of_neighbors_ID

        ###
        ### MERGE LOCAL CLUSTERS UNDER MUST-LINK CONSTRAINTS
        ###

        # Get the lists of data_IDs for which each point is in a Must-link constraint
        compatible_cluster_IDs: Dict[str, List[str]] = {
            data_ID_j: [
                data_ID_i
                for data_ID_i in self.list_of_data_IDs
                if (
                    self.constraints_manager.get_inferred_constraint(
                        data_ID1=data_ID_j,
                        data_ID2=data_ID_i,
                    )
                    == "MUST_LINK"
                )
            ]
            for data_ID_j in self.list_of_data_IDs
        }

        # Get the lists of local clusters where each point is in
        clusters_of_data_IDs: Dict[str, List[str]] = {
            data_ID_j: [
                cluster_id
                for cluster_id in self.dict_of_local_clusters.keys()
                if (data_ID_j in self.dict_of_local_clusters[cluster_id])
            ]
            for data_ID_j in self.list_of_data_IDs
        }

        # Initialize a variable in order to analyze a point Must-link constraints only once
        list_of_analyzed_IDs: List[str] = []

        # Initialize a variable in order not to take one point into account in several core local clusters
        dict_of_assigned_local_cluster: Dict[str, str] = {data_ID: "NONE" for data_ID in self.list_of_data_IDs}

        for data_ID_i in self.list_of_data_IDs:
            if data_ID_i not in list_of_analyzed_IDs:
                if compatible_cluster_IDs[data_ID_i]:
                    # Choose a coherent ID of core local cluster corresponding to a local cluster ID of data_ID_i

                    # Initialize ID of the potential local cluster of data_ID_i and list of involved points
                    local_cluster_i_points: List[str] = []

                    if self.dict_of_data_IDs_labels[data_ID_i] == "NOISE":
                        data_ID_i_cluster = data_ID_i
                        local_cluster_i_points = [data_ID_i]

                    elif data_ID_i in self.dict_of_local_clusters.keys():
                        data_ID_i_cluster = data_ID_i
                        local_cluster_i_points = self.dict_of_local_clusters[data_ID_i]

                    else:
                        # Choose a local cluster ID where data_ID_i is in,
                        # and preferably a local cluster ID that is not already in a core local cluster

                        data_ID_i_cluster = clusters_of_data_IDs[data_ID_i][0]
                        for cluster_i_id in clusters_of_data_IDs[data_ID_i]:
                            if dict_of_assigned_local_cluster[data_ID_i] == "NONE":
                                data_ID_i_cluster = cluster_i_id
                                break
                        local_cluster_i_points = self.dict_of_local_clusters[data_ID_i_cluster]

                    for data_ID_j in compatible_cluster_IDs[data_ID_i]:
                        if self.dict_of_data_IDs_labels[data_ID_j] == "NOISE":
                            # Merge all the available points of the clusters involved in a Must-link constraint

                            list_of_core_cluster_points = []
                            for data_ID_k in local_cluster_i_points:
                                if dict_of_assigned_local_cluster[data_ID_k] == "NONE":
                                    list_of_core_cluster_points.append(data_ID_k)
                                    dict_of_assigned_local_cluster[data_ID_k] = data_ID_i_cluster

                            self.dict_of_core_local_clusters[data_ID_i_cluster] = list(
                                set(
                                    self.dict_of_core_local_clusters[data_ID_i_cluster]
                                    + list_of_core_cluster_points
                                    + [data_ID_i, data_ID_j]
                                )
                            )
                        else:
                            # Initialize ID of the potential local cluster of data_ID_j and the list of involved points
                            local_cluster_j_points = []

                            if data_ID_j in self.dict_of_local_clusters.keys():
                                local_cluster_j_points = [data_ID_j]

                            else:
                                # Choose a local cluster ID where data_ID_j is in,
                                # and preferably a local cluster ID that is not already in a core local cluster

                                data_ID_j_cluster = clusters_of_data_IDs[data_ID_j][0]
                                for cluster_j_id in clusters_of_data_IDs[data_ID_j]:
                                    if dict_of_assigned_local_cluster[data_ID_j] == "NONE":
                                        data_ID_j_cluster = cluster_j_id
                                        break
                                local_cluster_j_points = self.dict_of_local_clusters[data_ID_j_cluster]

                            # Merge all the available points of the clusters involved in a Must-link constraint

                            list_of_core_cluster_points = []
                            for data_ID_l in list(set(local_cluster_i_points + local_cluster_j_points)):
                                if dict_of_assigned_local_cluster[data_ID_l] == "NONE":
                                    list_of_core_cluster_points.append(data_ID_l)
                                    dict_of_assigned_local_cluster[data_ID_l] = data_ID_i_cluster

                            self.dict_of_core_local_clusters[data_ID_i_cluster] = list(
                                set(
                                    self.dict_of_core_local_clusters[data_ID_i_cluster]
                                    + list_of_core_cluster_points
                                    + [data_ID_i, data_ID_j]
                                )
                            )

                # Mark the current point as analyzed in order not to have it in two clusters
                list_of_analyzed_IDs.append(data_ID_i)

        # Clean the `dict_of_core_local_clusters` variable
        for data_ID in self.list_of_data_IDs:
            if not self.dict_of_core_local_clusters[data_ID]:
                # Clean by deleting non-existing core local clusters entries
                self.dict_of_core_local_clusters.pop(data_ID)
            elif dict_of_assigned_local_cluster[data_ID] != data_ID:
                # Clean by deleting core local clusters entries corresponding to another already created core cluster
                self.dict_of_core_local_clusters.pop(data_ID)

        # Clean the `dict_of_core_local_clusters` variable by removing single-point clusters
        # because don't make sense in a Must-link constraint
        for potential_single_data_ID in self.list_of_data_IDs:
            if (
                potential_single_data_ID in self.dict_of_core_local_clusters.keys()
                and len(self.dict_of_core_local_clusters[potential_single_data_ID]) < 2
            ):
                self.dict_of_core_local_clusters.pop(potential_single_data_ID)

        ###
        ### MERGE LOCAL CLUSTERS UNDER CANNOT-LINK CONSTRAINTS
        ###

        for core_cluster_ID in self.dict_of_core_local_clusters.keys():
            merging = True

            while merging and self.dict_of_local_clusters:
                # While there is no conflict and there is still local clusters

                distances_to_local_clusters: Dict[str, float] = {}

                # Compute the distances between the core cluster and the local clusters
                for local_cluster_ID in self.dict_of_local_clusters.keys():
                    # Compute the smallest distance between points of the core cluster and the local cluster
                    distances_to_local_clusters[local_cluster_ID] = min(
                        [
                            self.dict_of_pairwise_distances[core_cluster_pt][local_cluster_pt]
                            for core_cluster_pt in self.dict_of_core_local_clusters[core_cluster_ID]
                            for local_cluster_pt in self.dict_of_local_clusters[local_cluster_ID]
                        ]
                    )

                # Find closest local cluster to core cluster
                closest_cluster = min(
                    distances_to_local_clusters
                )  # TODO: min(distances_to_local_clusters, key=lambda x: distances_to_local_clusters[x])

                if distances_to_local_clusters[closest_cluster] > self.eps:
                    merging = False

                else:
                    # Get the lists of not compatible data_IDs for deciding if clusters are merged
                    not_compatible_IDs: List[List[str]] = [
                        [
                            data_ID_m
                            for data_ID_m in self.dict_of_local_clusters[closest_cluster]
                            if (
                                self.constraints_manager.get_inferred_constraint(
                                    data_ID1=data_ID_n,
                                    data_ID2=data_ID_m,
                                )
                                == "CANNOT_LINK"
                            )
                        ]
                        for data_ID_n in self.dict_of_core_local_clusters[core_cluster_ID]
                    ]

                    # Check if there is a Cannot-link constraint between the points
                    no_conflict = True
                    for core_local_cluster_not_compatible_IDs in not_compatible_IDs:
                        if core_local_cluster_not_compatible_IDs:
                            no_conflict = False
                            break

                    if no_conflict:
                        # Merge core local cluster and its closest local cluster
                        self.dict_of_core_local_clusters[core_cluster_ID] = list(
                            set(
                                self.dict_of_core_local_clusters[core_cluster_ID]
                                + self.dict_of_local_clusters[closest_cluster]
                            )
                        )

                        self.dict_of_local_clusters.pop(closest_cluster)

                    else:
                        merging = False

        ###
        ### DEFINING FINAL CLUSTERS
        ###

        # Consider the final core local clusters
        assigned_cluster_id: int = 0
        for core_cluster in self.dict_of_core_local_clusters.keys():
            for cluster_point in self.dict_of_core_local_clusters[core_cluster]:
                self.dict_of_predicted_clusters[cluster_point] = assigned_cluster_id
            assigned_cluster_id += 1

        # Consider the remaining local clusters
        for local_cluster in self.dict_of_local_clusters.keys():
            # Remove points that already are in a final cluster
            points_to_remove = []
            for local_cluster_point in self.dict_of_local_clusters[local_cluster]:
                if local_cluster_point in self.dict_of_predicted_clusters.keys():
                    points_to_remove.append(local_cluster_point)
            for data_ID_to_remove in points_to_remove:
                self.dict_of_local_clusters[local_cluster].remove(data_ID_to_remove)

            # Check that the local cluster is still big enough
            if len(self.dict_of_local_clusters[local_cluster]) >= self.eps:
                for core_cluster_point in self.dict_of_local_clusters[local_cluster]:
                    self.dict_of_predicted_clusters[core_cluster_point] = assigned_cluster_id
                assigned_cluster_id += 1

        # Rename clusters
        self.dict_of_predicted_clusters = rename_clusters_by_order(
            clusters=self.dict_of_predicted_clusters,
        )

        # Set number of regular clusters
        self.number_of_regular_clusters = np.unique(np.array(list(self.dict_of_predicted_clusters.values()))).shape[0]

        # Consider ignored points
        ignored_cluster_id: int = -1
        for potential_ignored_point in self.list_of_data_IDs:
            if potential_ignored_point not in self.dict_of_predicted_clusters:
                self.dict_of_predicted_clusters[potential_ignored_point] = ignored_cluster_id
                ignored_cluster_id -= 1

        # Set number of single ignored points cluster
        self.number_of_single_noise_point_clusters = -(ignored_cluster_id + 1)

        # Set total number of clusters
        self.number_of_clusters = self.number_of_regular_clusters + self.number_of_single_noise_point_clusters

        return self.dict_of_predicted_clusters
