# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.abstract
* Description:  The abstract class used to define constrained clustering algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from abc import ABC, abstractmethod  # To define an abstract class.
from typing import Dict  # To type Python code (mypy).

from scipy.sparse import csr_matrix  # To handle matrix and vectors.

from cognitivefactory.interactive_clustering.constraints.abstract import (  # To manage constraints.
    AbstractConstraintsManager,
)


# ==============================================================================
# ABSTRACT CONSTRAINED CLUSTERING
# ==============================================================================
class AbstractConstrainedClustering(ABC):
    """
    Abstract class that is used to define constrained clustering algorithms.
    The main inherited method is `cluster`.

    References:
        - Survey on Constrained Clustering : `Lampert, T., T.-B.-H. Dao, B. Lafabregue, N. Serrette, G. Forestier, B. Cremilleux, C. Vrain, et P. Gancarski (2018). Constrained distance based clustering for time-series : a comparative and experimental study. Data Mining and Knowledge Discovery 32(6), 1663–1707.`
    """

    # ==============================================================================
    # ABSTRACT METHOD - CLUSTER
    # ==============================================================================
    @abstractmethod
    def cluster(
        self,
        constraints_manager: AbstractConstraintsManager,
        vectors: Dict[str, csr_matrix],
        nb_clusters: int,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to cluster data.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs that will force clustering to respect some conditions during computation.
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data.
            nb_clusters (int): The number of clusters to compute. #TODO Set defaults to None with elbow method or other method ?
            verbose (bool, optional): Enable verbose output. Defaults to `False`.
            **kargs (dict): Other parameters that can be used in the clustering.

        Raises:
            ValueError: if `vectors` and `constraints_manager` are incompatible, or if some parameters are incorrectly set.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """


# ==============================================================================
# RENAME CLUSTERS BY ORDER
# ==============================================================================
def rename_clusters_by_order(
    clusters: Dict[str, int],
) -> Dict[str, int]:
    """
    Rename cluster ID to be ordered by data IDs.

    Args:
        clusters (Dict[str, int]): The dictionary of clusters.

    Returns:
        Dict[str, int]: The sorted dictionary of clusters.
    """

    # Get `list_of_data_IDs`.
    list_of_data_IDs = sorted(clusters.keys())

    # Define a map to be able to rename cluster IDs.
    mapping_of_old_ID_to_new_ID: Dict[int, int] = {}
    new_ID: int = 0
    for data_ID in list_of_data_IDs:  # , cluster_ID in clusters.items():
        if clusters[data_ID] not in mapping_of_old_ID_to_new_ID.keys():
            mapping_of_old_ID_to_new_ID[clusters[data_ID]] = new_ID
            new_ID += 1

    # Rename cluster IDs.
    new_clusters = {
        data_ID_to_assign: mapping_of_old_ID_to_new_ID[clusters[data_ID_to_assign]]
        for data_ID_to_assign in list_of_data_IDs
    }

    # Return the new ordered clusters
    return new_clusters
