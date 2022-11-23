# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_dbscan.py
* Description:  Unittests for the `clustering.dbscan` module.
* Author:       David Nicolazo
* Created:      2/11/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================


import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
from cognitivefactory.interactive_clustering.clustering.affinity_propagation import AffinityPropagationConstrainedClustering



# ==============================================================================
# test_DBScanConstrainedClustering_cluster_with_no_constraints_1
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_with_no_constraints_1():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` clustering works with no `constraints`.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "0": csr_matrix([1.00, 0.00, 0.00, 0.00]),
        "1": csr_matrix([0.00, 0.43, 0.00, 0.00]),
        "2": csr_matrix([0.00, 0.00, 0.29, 0.00]),
        "3": csr_matrix([0.00, 0.00, 0.50, 0.00]),
        "4": csr_matrix([0.00, 0.00, 0.00, 0.98]),
        "5": csr_matrix([0.00, 0.00, 0.33, 0.00]),
        "6": csr_matrix([0.00, 0.00, 0.00, 1.40]),
        "7": csr_matrix([0.80, 0.00, 0.00, 0.00]),
        "8": csr_matrix([0.00, 0.54, 0.00, 0.00]),
        "9": csr_matrix([0.00, 0.00, 0.00, 1.10]),
        "10": csr_matrix([1.10, 0.00, 0.00, 0.00]),
        "11": csr_matrix([0.00, 0.49, 0.00, 0.00]),
    }

    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

    print("AP1", dict_of_predicted_clusters)

    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 2,
        "4": 3,
        "5": 2,
        "6": 3,
        "7": 0,
        "8": 1,
        "9": 3,
        "10": 0,
        "11": 1,
    }


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_with_no_constraints_2
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_with_no_constraints_2():

    # Define `vectors` and `constraints_manager`
    vectors = {
        "0": csr_matrix([2.00, 0.00, 0.00, 0.00]),
        "1": csr_matrix([0.00, 0.43, 0.00, 0.00]),
        "2": csr_matrix([0.00, 0.00, 0.29, 0.00]),
        "3": csr_matrix([0.00, 0.00, 0.50, 0.00]),
        "4": csr_matrix([0.00, 0.00, 0.00, 0.98]),
        "5": csr_matrix([0.00, 0.00, 0.33, 0.00]),
        "6": csr_matrix([0.00, 0.00, 0.00, 1.40]),
        "7": csr_matrix([0.80, 0.00, 0.00, 0.00]),
        "8": csr_matrix([0.00, 0.54, 0.00, 0.00]),
        "9": csr_matrix([0.00, 0.00, 0.00, 1.10]),
        "10": csr_matrix([1.10, 0.00, 0.00, 0.00]),
        "11": csr_matrix([0.00, 0.49, 0.00, 0.00]),
    }

    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

    print("AP2", dict_of_predicted_clusters)

    assert clustering_model.dict_of_predicted_clusters

    """
    Here, '0' is too far from other points so it is noise
    Furthermore, '7' and '10' are in the same neighbourhood, but no other point.
    They are not numerous enough to create a cluster
    """

    assert dict_of_predicted_clusters == {"1": 0, "2": 1, "3": 1, "4": 2, "5": 1, "6": 2, "8": 0, "9": 2, "11": 0}


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_with_some_constraints
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_with_some_constraints():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` clustering works with no `constraints`.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "0": csr_matrix([2.00, 0.00, 0.00, 0.00]),
        "1": csr_matrix([0.00, 0.43, 0.00, 0.00]),
        "2": csr_matrix([0.00, 0.00, 0.29, 0.00]),
        "3": csr_matrix([0.00, 0.00, 0.50, 0.00]),
        "4": csr_matrix([0.00, 0.00, 0.00, 0.98]),
        "5": csr_matrix([0.00, 0.00, 0.33, 0.00]),
        "6": csr_matrix([0.00, 0.00, 0.00, 1.40]),
        "7": csr_matrix([0.80, 0.00, 0.00, 0.00]),
        "8": csr_matrix([0.00, 0.54, 0.00, 0.00]),
        "9": csr_matrix([0.00, 0.00, 0.00, 1.10]),
        "10": csr_matrix([1.10, 0.00, 0.00, 0.00]),
        "11": csr_matrix([0.00, 0.49, 0.00, 0.00]),
    }

    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))
    constraints_manager.add_constraint(data_ID1="0", data_ID2="7", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="10", constraint_type="MUST_LINK")

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

    print("AP3", dict_of_predicted_clusters)

    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 2,
        "4": 3,
        "5": 2,
        "6": 3,
        "7": 0,
        "8": 1,
        "9": 3,
        "10": 0,
        "11": 1,
    }
