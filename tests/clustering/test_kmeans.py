# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_kmeans.py
* Description:  Unittests for the `clustering.kmeans` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.clustering.kmeans import KMeansConstrainedClustering
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# test_KMeansConstrainedClustering_for_inconsistent_model
# ==============================================================================
def test_KMeansConstrainedClustering_for_inconsistent_model():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` initialization raises an `ValueError` for inconsistent `model` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`model`"):
        KMeansConstrainedClustering(
            model="as_you_want",
        )


# ==============================================================================
# test_KMeansConstrainedClustering_for_inconsistent_max_iteration
# ==============================================================================
def test_KMeansConstrainedClustering_for_inconsistent_max_iteration():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` initialization raises an `ValueError` for inconsistent `max_iteration` parameter.
    """

    # Check `ValueError` for bad string value for `max_iteration`.
    with pytest.raises(ValueError, match="`max_iteration`"):
        KMeansConstrainedClustering(
            max_iteration=-1,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_for_inconsistent_tolerance
# ==============================================================================
def test_KMeansConstrainedClustering_for_inconsistent_tolerance():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` initialization raises an `ValueError` for inconsistent `tolerance` parameter.
    """

    # Check `ValueError` for bad string value for `tolerance`.
    with pytest.raises(ValueError, match="`tolerance`"):
        KMeansConstrainedClustering(
            tolerance=-1,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_for_correct_settings
# ==============================================================================
def test_KMeansConstrainedClustering_for_correct_settings():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` initialization runs correctly with the correct settings.
    """

    # Check a correct initialization.
    clustering_model = KMeansConstrainedClustering(
        model="COP",
        max_iteration=100,
        tolerance=1e-3,
        random_seed=3,
    )
    assert clustering_model
    assert clustering_model.model == "COP"
    assert clustering_model.max_iteration == 100
    assert math.isclose(clustering_model.tolerance, 1e-3)
    assert clustering_model.random_seed == 3


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_for_inconsistent_constraints_manager
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_for_inconsistent_constraints_manager():
    """
    Test that the `clustering.spectral.KMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints_manager` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            constraints_manager=None,
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.spectral.KMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters():
    """
    Test that the `clustering.spectral.KMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=-1,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_with_no_constraints_1
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_no_constraints_1():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with no `constraints`.
    """

    # Define `vectors` and `constraints_manager`
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=2,
    )

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=2,
    )

    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 1, "7": 1, "8": 1}


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_with_no_constraints_2
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_no_constraints_2():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with no `constraints`.
    """

    # Define `vectors` and `constraints_manager`
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=2,
    )

    # Run clustering 3 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=3,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 1,
        "5": 1,
        "6": 1,
        "7": 2,
        "8": 2,
        "9": 2,
    }


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_with_some_constraints
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_some_constraints():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with some `constraints`.
    """

    # Define `vectors` and `constraints_manager`
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="7", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="8", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="5", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="4", constraint_type="CANNOT_LINK")

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=2,
    )

    # Run clustering 2 clusters and somme constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=3,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 1,
        "4": 2,
        "5": 2,
        "6": 0,
        "7": 0,
        "8": 0,
    }


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_with_full_constraints
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_full_constraints():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with full `constraints`.
    """

    # Define `vectors` and `constraints_manager`
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="8", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="5", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="6", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="3", data_ID2="7", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="2", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="3", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="2", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="3", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="3", constraint_type="CANNOT_LINK")

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Run clustering 4 clusters and full constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=4,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 0,
        "5": 1,
        "6": 2,
        "7": 3,
        "8": 0,
    }


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_with_no_possible_cluster
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_no_possible_cluster():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with no possible cluster.
    """

    # Define `vectors` and `constraints_manager`
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="2", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="3", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="5", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="6", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="2", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="3", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="5", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="6", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="1", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="3", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="5", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="6", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="3", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="3", data_ID2="5", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="3", data_ID2="6", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="3", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="3", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="5", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="6", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="6", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="6", data_ID2="7", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="6", data_ID2="8", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="7", data_ID2="8", constraint_type="CANNOT_LINK")

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=3,
    )

    # Run clustering.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=5,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
    }


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_with_max_iteration_ending
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_max_iteration_ending():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with `max_iteration` ending.
    """

    # Define `vectors` and `constraints_manager`
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        max_iteration=1,
    )

    # Run clustering.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=2,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters
