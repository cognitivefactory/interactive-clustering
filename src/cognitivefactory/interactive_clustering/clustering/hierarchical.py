# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.hierarchical
* Description:  Implementation of constrained hierarchical clustering algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from datetime import datetime  # To use date in verbose output.
from typing import Any, Dict, List, Optional, Tuple  # To type Python code (mypy).

from scipy.sparse import csr_matrix, vstack  # To handle matrix and vectors.
from sklearn.metrics import pairwise_distances  # To compute distance.

from cognitivefactory.interactive_clustering.clustering.abstract import (  # To use abstract interface.; To sort clusters after computation.
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)
from cognitivefactory.interactive_clustering.constraints.abstract import (  # To manage constraints.
    AbstractConstraintsManager,
)


# ==============================================================================
# HIERARCHICAL CONSTRAINED CLUSTERING
# ==============================================================================
class HierarchicalConstrainedClustering(AbstractConstrainedClustering):
    """
    This class implements the hierarchical constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    References:
        - Hierarchical Clustering: `Murtagh, F. et P. Contreras (2012). Algorithms for hierarchical clustering : An overview. Wiley Interdisc. Rew.: Data Mining and Knowledge Discovery 2, 86–97.`
        - Constrained Hierarchical Clustering: `Davidson, I. et S. S. Ravi (2005). Agglomerative Hierarchical Clustering with Constraints : Theoretical and Empirical Results. Springer, Berlin, Heidelberg 3721, 12.`

    Examples:
        ```python
        # Import.
        from scipy.sparse import csr_matrix
        from cognitivefactory.interactive_clustering.clustering.hierarchical import HierarchicalConstrainedClustering

        # Create an instance of hierarchical clustering.
        clustering_model = HierarchicalConstrainedClustering(
            linkage="ward",
            random_seed=2,
        )

        # Define vectors.
        # NB : use cognitivefactory.interactive_clustering.utils to preprocess and vectorize texts.
        vectors = {
            "0": csr_matrix([1.00, 0.00, 0.00]),
            "1": csr_matrix([0.95, 0.02, 0.01]),
            "2": csr_matrix([0.98, 0.00, 0.00]),
            "3": csr_matrix([0.99, 0.00, 0.00]),
            "4": csr_matrix([0.01, 0.99, 0.07]),
            "5": csr_matrix([0.02, 0.99, 0.07]),
            "6": csr_matrix([0.01, 0.99, 0.02]),
            "7": csr_matrix([0.01, 0.01, 0.97]),
            "8": csr_matrix([0.00, 0.01, 0.99]),
            "9": csr_matrix([0.00, 0.00, 1.00]),
        }

        # Define constraints manager.
        constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

        # Run clustering.
        dict_of_predicted_clusters = clustering_model(
            constraints_manager=constraints_manager,
            vectors=vectors,
            nb_clusters=3,
        )

        # Print results.
        print("Expected results", ";", {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1, "7": 2, "8": 2, "9": 2,})
        print("Computed results", ":", dict_of_predicted_clusters)
        ```
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(self, linkage: str = "ward", random_seed: Optional[int] = None, **kargs) -> None:
        """
        The constructor for Hierarchical Constrainted Clustering class.

        Args:
            linkage (str, optional): The metric used to merge clusters. Several type are implemented :
                - `"ward"`: Merge the two clusters for which the merged cluster from these clusters have the lowest intra-class distance.
                - `"average"`: Merge the two clusters that have the closest barycenters.
                - `"complete"`: Merge the two clusters for which the maximum distance between two data of these clusters is the lowest.
                - `"single"`: Merge the two clusters for which the minimum distance between two data of these clusters is the lowest.
                Defaults to `"ward"`.
            random_seed (Optional[int], optional): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Store `self.linkage`.
        if linkage not in {"ward", "average", "complete", "single"}:
            raise ValueError("The `linkage` '" + str(linkage) + "' is not implemented.")
        self.linkage: str = linkage

        # Store `self.random_seed`
        self.random_seed: Optional[int] = random_seed

        # Store `self.kargs` for hierarchical clustering.
        self.kargs = kargs

        # Initialize `self.clustering_root` and `self.dict_of_predicted_clusters`.
        self.clustering_root: Optional[Cluster] = None
        self.dict_of_predicted_clusters: Optional[Dict[str, int]] = None

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================
    def cluster(
        self,
        constraints_manager: AbstractConstraintsManager,
        vectors: Dict[str, csr_matrix],
        nb_clusters: int,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the Hierarchical model.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs that will force clustering to respect some conditions during computation.
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data.
            nb_clusters (int): The number of clusters to compute.  #TODO Set defaults to None with elbow method or other method ?
            verbose (bool, optional): Enable verbose output. Defaults to `False`.
            **kargs (dict): Other parameters that can be used in the clustering.

        Raises:
            ValueError: If some parameters are incorrectly set.

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
        if nb_clusters < 2:
            raise ValueError("The `nb_clusters` '" + str(nb_clusters) + "' must be greater than or equal to 2.")
        self.nb_clusters: int = nb_clusters

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
        ### INITIALIZE HIERARCHICAL CONSTRAINED CLUSTERING
        ###

        # Verbose
        if verbose:  # pragma: no cover

            # Verbose - Print progression status.
            TIME_start: datetime = datetime.now()
            print(
                "    ",
                "CLUSTERING_ITERATION=" + "INITIALIZATION",
                "(current_time = " + str(TIME_start - TIME_start).split(".")[0] + ")",
            )

        # Initialize `self.clustering_root` and `self.dict_of_predicted_clusters`.
        self.clustering_root = None
        self.dict_of_predicted_clusters = None

        # Initialize iteration counter.
        self.clustering_iteration: int = 0

        # Initialize `current_clusters` and `self.clusters_storage`.
        self.current_clusters: List[int] = []
        self.clusters_storage: Dict[int, Cluster] = {}

        # Get the list of possibles lists of MUST_LINK data for initialization.
        list_of_possible_lists_of_MUST_LINK_data: List[List[str]] = self.constraints_manager.get_connected_components()

        # Estimation of max number of iteration.
        max_clustering_iteration: int = len(list_of_possible_lists_of_MUST_LINK_data) - 1

        # For each list of same data (MUST_LINK constraints).
        for MUST_LINK_data in list_of_possible_lists_of_MUST_LINK_data:

            # Create a initial cluster with data that MUST be LINKed.
            self._add_new_cluster_by_setting_members(
                members=MUST_LINK_data,
            )

        # Initialize distance between clusters.
        self.clusters_distance: Dict[int, Dict[int, float]] = {}
        for cluster_IDi in self.current_clusters:
            for cluster_IDj in self.current_clusters:
                if cluster_IDi < cluster_IDj:
                    # Compute distance between cluster i and cluster j.
                    distance: float = self._compute_distance(cluster_IDi=cluster_IDi, cluster_IDj=cluster_IDj)
                    # Store distance between cluster i and cluster j.
                    self._set_distance(cluster_IDi=cluster_IDi, cluster_IDj=cluster_IDj, distance=distance)

        # Initialize iterations at first iteration.
        self.clustering_iteration = 1

        ###
        ### RUN ITERATIONS OF HIERARCHICAL CONSTRAINED CLUSTERING UNTIL CONVERGENCE
        ###

        # Iter until convergence of clustering.
        while len(self.current_clusters) > 1:

            # Verbose
            if verbose:  # pragma: no cover

                # Verbose - Print progression status.
                TIME_current: datetime = datetime.now()
                print(
                    "    ",
                    "CLUSTERING_ITERATION="
                    + str(self.clustering_iteration).zfill(6)
                    + "/"
                    + str(max_clustering_iteration).zfill(6),
                    "(current_time = " + str(TIME_current - TIME_start).split(".")[0] + ")",
                    end="\r",
                )

            # Get clostest clusters to merge
            clostest_clusters: Optional[Tuple[int, int]] = self._get_the_two_clostest_clusters()

            # If no clusters to merge, then stop iterations.
            if clostest_clusters is None:
                break

            # Merge clusters the two closest clusters and add the merged cluster to the storage.
            # If merge one cluster "node" with a cluster "leaf" : add the cluster "leaf" to the children of the cluster "node".
            # If merge two clusters "nodes" or two clusters "leaves" : create a new cluster "node".
            merged_cluster_ID: int = self._add_new_cluster_by_merging_clusters(
                children=[
                    clostest_clusters[0],
                    clostest_clusters[1],
                ]
            )

            # Update distances
            for cluster_ID in self.current_clusters:

                if cluster_ID != merged_cluster_ID:
                    # Compute distance between cluster and merged cluster.
                    distance = self._compute_distance(cluster_IDi=cluster_ID, cluster_IDj=merged_cluster_ID)
                    # Store distance between cluster and merged cluster.
                    self._set_distance(cluster_IDi=cluster_ID, cluster_IDj=merged_cluster_ID, distance=distance)

            # Update self.clustering_iteration.
            self.clustering_iteration += 1

        ###
        ### END HIERARCHICAL CONSTRAINED CLUSTERING
        ###

        # Verbose
        if verbose:  # pragma: no cover

            # Verbose - Print progression status.
            TIME_current = datetime.now()

            # Case of clustering not completed.
            if len(self.current_clusters) > 1:
                print(
                    "    ",
                    "CLUSTERING_ITERATION=" + str(self.clustering_iteration).zfill(5),
                    "-",
                    "End : No more cluster to merge",
                    "(current_time = " + str(TIME_current - TIME_start).split(".")[0] + ")",
                )
            else:
                print(
                    "    ",
                    "CLUSTERING_ITERATION=" + str(self.clustering_iteration).zfill(5),
                    "-",
                    "End : Full clustering done",
                    "(current_time = " + str(TIME_current - TIME_start).split(".")[0] + ")",
                )

        # If several clusters remains, then merge them in a cluster root.
        if len(self.current_clusters) > 1:

            # Merge all remaining clusters.
            # If merge one cluster "node" with many cluster "leaves" : add clusters "leaves" to the children of the cluster "node".
            # If merge many clusters "nodes" and/or many clusters "leaves" : create a new cluster "node".
            self._add_new_cluster_by_merging_clusters(children=self.current_clusters.copy())

        # Get clustering root.
        root_ID: int = self.current_clusters[0]
        self.clustering_root = self.clusters_storage[root_ID]

        ###
        ### GET PREDICTED CLUSTERS
        ###

        # Compute predicted clusters.
        self.dict_of_predicted_clusters = self.compute_predicted_clusters(
            nb_clusters=self.nb_clusters,
        )

        return self.dict_of_predicted_clusters

    # ==============================================================================
    # ADD CLUSTER BY SETTING MEMBERS :
    # ==============================================================================
    def _add_new_cluster_by_setting_members(
        self,
        members: List[str],
    ) -> int:
        """
        Create or Update a cluster by setting its members, and add it to the storage and current clusters.

        Args:
            members (List[str]): A list of data IDs to define the new cluster by the data it contains.

        Returns:
            int : ID of the merged cluster.
        """
        # Get the ID of the new cluster.
        new_cluster_ID: int = max(self.clusters_storage.keys()) + 1 if (self.clusters_storage) else 0

        # Create the cluster.
        new_cluster = Cluster(
            vectors=self.vectors,
            cluster_ID=new_cluster_ID,
            clustering_iteration=self.clustering_iteration,
            members=members,
        )

        # Add new_cluster to `self.current_clusters` and `self.clusters_storage`.
        self.current_clusters.append(new_cluster_ID)
        self.clusters_storage[new_cluster_ID] = new_cluster

        return new_cluster_ID

    # ==============================================================================
    # ADD CLUSTER BY MERGING CLUSTERS :
    # ==============================================================================
    def _add_new_cluster_by_merging_clusters(
        self,
        children: List[int],
    ) -> int:
        """
        Create or Update a cluster by setting its children, and add it to the storage and current clusters.

        Args:
            children (List[int]): A list of cluster IDs to define the new cluster by its children.

        Returns:
            int : ID of the merged cluster.
        """

        # Remove all leaves children clusters from `self.current_clusters`.
        for child_ID_to_remove in children:
            self.current_clusters.remove(child_ID_to_remove)

        """
        ###
        ### Tree optimization : if only one node, then update this node as parent of all leaves.
        ### TODO : test of check if relevant to use. pros = smarter tree visualisation ; cons = cluster number more difficult to choose.
        ###

        # List of children nodes.
        list_of_children_nodes: List[int] = [
            child_ID
            for child_ID in children
            if len(self.clusters_storage[child_ID].children) > 0
        ]

        if len(list_of_children_nodes) == 1:

            # Get the ID of the cluster to update
            parent_cluster_ID: int = list_of_children_nodes[0]
            parent_cluster: Cluster = self.clusters_storage[parent_cluster_ID]

            # Add all leaves
            parent_cluster.add_new_children(
                new_children=[
                    self.clusters_storage[child_ID]
                    for child_ID in children
                    if child_ID != parent_cluster_ID
                ],
                new_clustering_iteration=self.clustering_iteration
            )

            # Add new_cluster to `self.current_clusters` and `self.clusters_storage`.
            self.current_clusters.append(parent_cluster_ID)
            self.clusters_storage[parent_cluster_ID] = parent_cluster

            # Return the cluster_ID of the created cluster.
            return parent_cluster_ID


        """

        ###
        ### Default case : Create a new node as parent of all children to merge.
        ###

        # Get the ID of the new cluster.
        parent_cluster_ID: int = max(self.clusters_storage) + 1

        # Create the cluster
        parent_cluster = Cluster(
            vectors=self.vectors,
            cluster_ID=parent_cluster_ID,
            clustering_iteration=self.clustering_iteration,
            children=[self.clusters_storage[child_ID] for child_ID in children],
        )

        # Add new_cluster to `self.current_clusters` and `self.clusters_storage`.
        self.current_clusters.append(parent_cluster_ID)
        self.clusters_storage[parent_cluster_ID] = parent_cluster

        # Return the cluster_ID of the created cluster.
        return parent_cluster_ID

    # ==============================================================================
    # COMPUTE DISTANCE BETWEEN CLUSTERING NEW ITERATION OF CLUSTERING :
    # ==============================================================================
    def _compute_distance(self, cluster_IDi: int, cluster_IDj: int) -> float:
        """
        Compute distance between two clusters.

        Args:
            cluster_IDi (int): ID of the first cluster.
            cluster_IDj (int): ID of the second cluster.

        Returns:
            float : Distance between the two clusters.
        """

        # Check `"CANNOT_LINK"` constraints.
        for data_ID1 in self.clusters_storage[cluster_IDi].members:
            for data_ID2 in self.clusters_storage[cluster_IDj].members:
                if (
                    self.constraints_manager.get_inferred_constraint(
                        data_ID1=data_ID1,
                        data_ID2=data_ID2,
                    )
                    == "CANNOT_LINK"
                ):
                    return float("Inf")

        # Case 1 : `self.linkage` is "complete".
        if self.linkage == "complete":

            return max(
                [
                    self.dict_of_pairwise_distances[data_ID_in_cluster_IDi][data_ID_in_cluster_IDj]
                    for data_ID_in_cluster_IDi in self.clusters_storage[cluster_IDi].members
                    for data_ID_in_cluster_IDj in self.clusters_storage[cluster_IDj].members
                ]
            )

        # Case 2 : `self.linkage` is "average".
        if self.linkage == "average":
            return pairwise_distances(
                X=self.clusters_storage[cluster_IDi].centroid,
                Y=self.clusters_storage[cluster_IDj].centroid,
                metric="euclidean",  # TODO: Load different parameters for distance computing ?
            )[0][0]

        # Case 3 : `self.linkage` is "single".
        if self.linkage == "single":
            return min(
                [
                    self.dict_of_pairwise_distances[data_ID_in_cluster_IDi][data_ID_in_cluster_IDj]
                    for data_ID_in_cluster_IDi in self.clusters_storage[cluster_IDi].members
                    for data_ID_in_cluster_IDj in self.clusters_storage[cluster_IDj].members
                ]
            )

        # Case 4 : `self.linkage` is "ward".
        ##if self.linkage == "ward": ## DEFAULTS
        # Compute distance
        merged_members: List[str] = (
            self.clusters_storage[cluster_IDi].members + self.clusters_storage[cluster_IDj].members
        )
        return sum(
            [
                self.dict_of_pairwise_distances[data_IDi][data_IDj]
                for i, data_IDi in enumerate(merged_members)
                for j, data_IDj in enumerate(merged_members)
                if i < j
            ]
        ) / (len(self.clusters_storage[cluster_IDi].members) * len(self.clusters_storage[cluster_IDj].members))

    # ==============================================================================
    # DISTANCE : GETTER
    # ==============================================================================
    def _get_distance(self, cluster_IDi: int, cluster_IDj: int) -> float:
        """
        Get the distance between two clusters.

        Args:
            cluster_IDi (int): ID of the first cluster.
            cluster_IDj (int): ID of the second cluster.

        Returns:
            float : Distance between the two clusters.
        """

        # Sort IDs of cluster.
        min_cluster_ID: int = min(cluster_IDi, cluster_IDj)
        max_cluster_ID: int = max(cluster_IDi, cluster_IDj)

        # Return the distance.
        return self.clusters_distance[min_cluster_ID][max_cluster_ID]

    # ==============================================================================
    # DISTANCE : SETTER
    # ==============================================================================
    def _set_distance(
        self,
        distance: float,
        cluster_IDi: int,
        cluster_IDj: int,
    ) -> None:
        """
        Set the distance between two clusters.

        Args:
            distance (float): The distance between the two clusters.
            cluster_IDi (int): ID of the first cluster.
            cluster_IDj (int): ID of the second cluster.
        """

        # Sort IDs of cluster.
        min_cluster_ID: int = min(cluster_IDi, cluster_IDj)
        max_cluster_ID: int = max(cluster_IDi, cluster_IDj)

        # Add distance to the dictionary of distance.
        if min_cluster_ID not in self.clusters_distance:
            self.clusters_distance[min_cluster_ID] = {}
        self.clusters_distance[min_cluster_ID][max_cluster_ID] = distance

    # ==============================================================================
    # GET THE TWO CLOSEST CLUSTERS
    # ==============================================================================
    def _get_the_two_clostest_clusters(self) -> Optional[Tuple[int, int]]:
        """
        Get the two clusters which are the two closest clusters.

        Returns:
            Optional(Tuple[int, int]) : The IDs of the two closest clusters to merge. Return None if no cluster is suitable.
        """

        # Compute the two clostest clusters to merge. take the closest distance, then the closest cluster size.
        clostest_clusters = min(
            [
                {
                    "cluster_ID1": cluster_ID1,
                    "cluster_ID2": cluster_ID2,
                    "distance": self._get_distance(cluster_IDi=cluster_ID1, cluster_IDj=cluster_ID2),
                    "merged_size": len(self.clusters_storage[cluster_ID1].members)
                    + len(self.clusters_storage[cluster_ID2].members)
                    # TODO : Choose between "distance then size(count)" and "size_type(boolean) then distance"
                }
                for cluster_ID1 in self.current_clusters
                for cluster_ID2 in self.current_clusters
                if cluster_ID1 < cluster_ID2
            ],
            key=lambda dst: (dst["distance"], dst["merged_size"]),
        )

        # Get clusters and distance.
        cluster_ID1: int = int(clostest_clusters["cluster_ID1"])
        cluster_ID2: int = int(clostest_clusters["cluster_ID2"])
        distance: float = clostest_clusters["distance"]

        # Check distance.
        if distance == float("Inf"):
            return None

        # Return the tow closest clusters.
        return cluster_ID1, cluster_ID2

    # ==============================================================================
    # COMPUTE PREDICTED CLUSTERS
    # ==============================================================================
    def compute_predicted_clusters(self, nb_clusters: int, by: str = "size") -> Dict[str, int]:
        """
        Compute the predicted clusters based on clustering tree and estimation of number of clusters.

        Args:
            nb_clusters (int): The number of clusters to compute.
            by (str, optional): A string to identifies the criteria used to explore `HierarchicalConstrainedClustering` tree. Can be `"size"` or `"iteration"`. Defaults to `"size"`.

        Raises:
            ValueError: if `clustering_root` was not set.

        Returns:
            Dict[str,int] : A dictionary that contains the predicted cluster for each data ID.
        """

        # Check that the clustering has been made.
        if self.clustering_root is None:
            raise ValueError("The `clustering_root` is not set, probably because clustering was not run.")

        ###
        ### EXPLORE CLUSTER TREE
        ###

        # Define the resulted list of children as the children of `HierarchicalConstrainedClustering` root.
        list_of_clusters: List[Cluster] = [self.clustering_root]

        # Explore `HierarchicalConstrainedClustering` children until dict_of_predicted_clusters has the right number of children.
        while len(list_of_clusters) < nb_clusters:

            if by == "size":
                # Get the biggest cluster in current children from `HierarchicalConstrainedClustering` exploration.
                # i.e. it's the cluster that has the more data to split.
                cluster_to_split = max(list_of_clusters, key=lambda c: len(c.members))

            else:  # if by == "iteration":
                # Get the most recent cluster in current children from `HierarchicalConstrainedClustering` exploration.
                # i.e. it's the cluster that was last merged.
                cluster_to_split = max(list_of_clusters, key=lambda c: c.clustering_iteration)

            # If the chosen cluster is a leaf : break the `HierarchicalConstrainedClustering` exploration.
            if cluster_to_split.children == []:  # noqa: WPS520
                break

            # Otherwise: The chosen cluster is a node, so split it and get its children.
            else:
                # ... remove the cluster obtained ...
                list_of_clusters.remove(cluster_to_split)

                # ... and add all its children.
                for child in cluster_to_split.children:
                    list_of_clusters.append(child)

        ###
        ### GET PREDICTED CLUSTERS
        ###

        # Initialize the dictionary of predicted clusters.
        predicted_clusters: Dict[str, int] = {data_ID: -1 for data_ID in self.list_of_data_IDs}

        # For all cluster...
        for cluster in list_of_clusters:

            # ... and for all member in each cluster...
            for data_ID in cluster.members:

                # ... affect the predicted cluster (cluster ID) to the data.
                predicted_clusters[data_ID] = cluster.cluster_ID

        # Rename cluster IDs by order.
        predicted_clusters = rename_clusters_by_order(clusters=predicted_clusters)

        # Return predicted clusters
        return predicted_clusters


# ==============================================================================
# CLUSTER
# ==============================================================================
class Cluster:
    """
    This class represents a cluster as a node of the hierarchical clustering tree.
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(
        self,
        vectors: Dict[str, csr_matrix],
        cluster_ID: int,
        clustering_iteration: int,
        children: Optional[List["Cluster"]] = None,
        members: Optional[List[str]] = None,
    ) -> None:
        """
        The constructor for Cluster class.

        Args:
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager` (if `constraints_manager` is set). The value of the dictionary represent the vector of each data.
            cluster_ID (int): The cluster ID that is defined during `HierarchicalConstrainedClustering.cluster` running.
            clustering_iteration (int): The cluster iteration that is defined during `HierarchicalConstrainedClustering.cluster` running.
            children (Optional[List["Cluster"]], optional): A list of clusters children for cluster initialization. Incompatible with `members` parameter. Defaults to `None`.
            members (Optional[List[str]], optional): A list of data IDs for cluster initialization. Incompatible with `children` parameter. Defaults to `None`.

        Raises:
            ValueError: if `children` and `members` are both set or both unset.
        """

        # Store links to `vectors`.
        self.vectors: Dict[str, csr_matrix] = vectors

        # Cluster ID and Clustering iteration.
        self.cluster_ID: int = cluster_ID
        self.clustering_iteration: int = clustering_iteration

        # Check children and members.
        if ((children is not None) and (members is not None)) or ((children is None) and (members is None)):
            raise ValueError(
                "Cluster initialization must be by `children` setting or by `members` setting, but not by both or none of them."
            )

        # Add children (empty or not).
        self.children: List["Cluster"] = children if (children is not None) else []

        # Cluster inverse depth.
        self.cluster_inverse_depth: int = (
            max([child.cluster_inverse_depth for child in self.children]) + 1 if (self.children) else 0
        )

        # Add members (empty or not).
        self.members: List[str] = (
            members if members is not None else [data_ID for child in self.children for data_ID in child.members]
        )

        # Update centroids
        self.update_centroid()

    # ==============================================================================
    # ADD NEW CHILDREN :
    # ==============================================================================
    def add_new_children(
        self,
        new_children: List["Cluster"],
        new_clustering_iteration: int,
    ) -> None:
        """
        Add new children to the cluster.

        Args:
            new_children (List["Cluster"]): The list of new clusters children to add.
            new_clustering_iteration (int): The new cluster iteration that is defined during HierarchicalConstrainedClustering.clusterize running.
        """

        # Update clustering iteration.
        self.clustering_iteration = new_clustering_iteration

        # Update children.
        self.children += [new_child for new_child in new_children if new_child not in self.children]

        # Update cluster inverse depth.
        self.cluster_inverse_depth = max([child.cluster_inverse_depth for child in self.children]) + 1

        # Update members.
        self.members = [data_ID for child in self.children for data_ID in child.members]

        # Update centroids.
        self.update_centroid()

    # ==============================================================================
    # UPDATE CENTROIDS :
    # ==============================================================================
    def update_centroid(self) -> None:
        """
        Update centroid of the cluster.
        """

        # Update centroids.
        self.centroid: csr_matrix = sum([self.vectors[data_ID] for data_ID in self.members]) / self.get_cluster_size()

    # ==============================================================================
    # GET CLUSTER SIZE :
    # ==============================================================================
    def get_cluster_size(self) -> int:
        """
        Get cluster size.

        Returns:
            int: The cluster size, i.e. the number of members in the cluster.
        """

        # Update centroids.
        return len(self.members)

    # ==============================================================================
    # TO DICTIONARY :
    # ==============================================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Transform the Cluster object into a dictionary. It can be used before serialize this object in JSON.

        Returns:
            Dict[str, Any]: A dictionary that represents the Cluster object.
        """

        # Define the result dictionary.
        results: Dict[str, Any] = {}

        # Add clustering information.
        results["cluster_ID"] = self.cluster_ID
        results["clustering_iteration"] = self.clustering_iteration

        # Add children information.
        results["children"] = [child.to_dict() for child in self.children]
        results["cluster_inverse_depth"] = self.cluster_inverse_depth

        # Add members information.
        results["members"] = self.members

        return results
