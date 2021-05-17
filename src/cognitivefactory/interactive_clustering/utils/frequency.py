# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.utils.frequency
* Description:  Utilities methods for frequency analysis.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from typing import Dict, List  # To type Python code (mypy).


# ==============================================================================
# COMPUTE FREQUENCY OF CLUSTERS :
# ==============================================================================
def compute_clusters_frequency(clustering_result: Dict[str, int]) -> Dict[int, float]:
    """
    Get the frequency of each cluster present in a clustering result.

    Args:
        clustering_result (Dict[str,int]): The dictionary that contains the predicted cluster for each data ID.

    Returns:
        Dict[int,float] : Frequency fo each predicted intent.
    """

    # Get the total number of data IDs.
    nb_of_data_IDs = len(clustering_result.keys())

    # Default case : No data, so no cluster.
    if nb_of_data_IDs == 0:
        return {}

    # Get possible clusters IDs.
    list_of_possible_cluster_IDs: List[int] = sorted(
        {clustering_result[data_ID] for data_ID in clustering_result.keys()}
    )

    # Compute frequency of clusters in `clustering_result`.
    frequence_of_clusters: Dict[int, float] = {
        cluster_ID: len([data_ID for data_ID in clustering_result if clustering_result[data_ID] == cluster_ID])
        / nb_of_data_IDs
        for cluster_ID in list_of_possible_cluster_IDs
    }

    # Return the frequence of clusters.
    return frequence_of_clusters
