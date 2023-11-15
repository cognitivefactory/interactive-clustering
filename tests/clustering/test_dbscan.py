# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_dbscan.py
* Description:  Unittests for the `clustering.dbscan` module.
* Author:       Marc TRUTT, Esther LENOTRE, David NICOLAZO
* Created:      31/10/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
from src.cognitivefactory.interactive_clustering.clustering.dbscan import DBScanConstrainedClustering


# ==============================================================================
# test_DBScanConstrainedClustering_for_inconsistent_eps
# ==============================================================================
def test_DBScanConstrainedClustering_for_inconsistent_eps():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` initialization raises an `ValueError` for inconsistent `eps` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`eps`"):
        DBScanConstrainedClustering(
            eps=-1,
        )


# ==============================================================================
# test_DBScanConstrainedClustering_for_inconsistent_min_samples
# ==============================================================================
def test_DBScanConstrainedClustering_for_inconsistent_min_samples():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` initialization raises an `ValueError` for inconsistent `min_samples` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`min_samples`"):
        DBScanConstrainedClustering(
            min_samples=-1,
        )


# ==============================================================================
# test_DBScanConstrainedClustering_for_correct_settings
# ==============================================================================
def test_DBScanConstrainedClustering_for_correct_settings():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` initialization runs correctly with the correct settings.
    """

    # Check a correct initialization.
    clustering_model = DBScanConstrainedClustering(
        eps=0.5,
        min_samples=3,
    )
    assert clustering_model
    assert math.isclose(clustering_model.eps, 0.5)
    assert clustering_model.min_samples == 3


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_for_inconsistent_constraints_manager
# ==============================================================================
def test_DBScanConstrainedClustering_cluster_for_inconsistent_constraints_manager():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints_manager` parameter.
    """

    # Initialize a `DBScanConstrainedClustering` instance.
    clustering_model = DBScanConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            constraints_manager=None,
            vectors=None,
        )


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_DBScanConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `DBScanConstrainedClustering` instance.
    clustering_model = DBScanConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors=None,
        )


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_for_inconsistent_nb_clusters
# ==============================================================================
def test_DBScanConstrainedClustering_cluster_for_inconsistent_nb_clusters():
    """
    Test that the `clustering.dbscan.DBScanConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `DBScanConstrainedClustering` instance.
    clustering_model = DBScanConstrainedClustering()

    # Check `ValueError` for not matrix `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=4,
        )


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_with_no_constraints_1
# ==============================================================================
def test_DBScanConstrainedClustering_cluster_with_no_constraints_1():
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
    clustering_model = DBScanConstrainedClustering(eps=0.5, min_samples=3)

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

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
def test_DBScanConstrainedClustering_cluster_with_no_constraints_2():
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

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = DBScanConstrainedClustering(
        eps=0.5,
        min_samples=3,
    )

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

    assert clustering_model.dict_of_predicted_clusters

    """
    Here, '0' is too far from other points so it is noise
    Furthermore, '7' and '10' are in the same neighbourhood, but no other point.
    They are not numerous enough to create a cluster
    """

    assert dict_of_predicted_clusters == {
        "0": -1,
        "1": 0,
        "2": 1,
        "3": 1,
        "4": 2,
        "5": 1,
        "6": 2,
        "7": -2,
        "8": 0,
        "9": 2,
        "10": -3,
        "11": 0,
    }


# ==============================================================================
# test_DBScanConstrainedClustering_cluster_with_some_constraints
# ==============================================================================
def test_DBScanConstrainedClustering_cluster_with_some_constraints():
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
    constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = DBScanConstrainedClustering(eps=0.5, min_samples=3)

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

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
