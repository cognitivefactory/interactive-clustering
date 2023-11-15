# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_mpckmeans.py
* Description:  Unittests for the `clustering.mpckmeans` module.
* Author:       Esther LENOTRE, David NICOLAZO, Marc TRUTT
* Created:      02/11/2022
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
from src.cognitivefactory.interactive_clustering.clustering.mpckmeans import MPCKMeansConstrainedClustering


# ==============================================================================
# test_MPCKMeansConstrainedClustering_for_inconsistent_model
# ==============================================================================
def test_MPCKMeansConstrainedClustering_for_inconsistent_model():
    """
    Test that the `clustering.kmeans.MPCKMeansConstrainedClustering` initialization raises an `ValueError` for inconsistent `model` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`model`"):
        MPCKMeansConstrainedClustering(
            model="as_you_want",
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_for_inconsistent_max_iteration
# ==============================================================================
def test_MPCKMeansConstrainedClustering_for_inconsistent_max_iteration():
    """
    Test that the `clustering.kmeans.MPCKMeansConstrainedClustering` initialization raises an `ValueError` for inconsistent `max_iteration` parameter.
    """

    # Check `ValueError` for bad string value for `max_iteration`.
    with pytest.raises(ValueError, match="`max_iteration`"):
        MPCKMeansConstrainedClustering(
            max_iteration=-1,
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_for_inconsistent_w
# ==============================================================================
def test_MPCKMeansConstrainedClustering_for_inconsistent_w():
    """
    Test that the `clustering.kmeans.MPCKMeansConstrainedClustering` initialization raises an `ValueError` for inconsistent `w` parameter.
    """

    # Check `ValueError` for bad string value for `tolerance`.
    with pytest.raises(ValueError, match="`weight`"):
        MPCKMeansConstrainedClustering(
            w=-1,
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_for_correct_settings
# ==============================================================================
def test_MPCKMeansConstrainedClustering_for_correct_settings():
    """
    Test that the `clustering.kmeans.MPCKMeansConstrainedClustering` initialization runs correctly with the correct settings.
    """

    # Check a correct initialization.
    clustering_model = MPCKMeansConstrainedClustering(
        model="MPC",
        max_iteration=100,
        w=0.5,
        random_seed=3,
    )
    assert clustering_model
    assert clustering_model.model == "MPC"
    assert clustering_model.max_iteration == 100
    assert math.isclose(clustering_model.w, 0.5)
    assert clustering_model.random_seed == 3


# ==============================================================================
# test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_constraints_manager
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_constraints_manager():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints_manager` parameter.
    """

    # Initialize a `MPCKMeansConstrainedClustering` instance.
    clustering_model = MPCKMeansConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            constraints_manager=None,
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = MPCKMeansConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters_1
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters_1():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `MPCKMeansConstrainedClustering` instance.
    clustering_model = MPCKMeansConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=None,
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters_2
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters_2():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `MPCKMeansConstrainedClustering` instance.
    clustering_model = MPCKMeansConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=-1,
        )


# ==============================================================================
# test_MPCKMeansConstrainedClustering_cluster_with_no_constraints_1
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_with_no_constraints_1():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering works with no `constraints`.
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

    # Initialize a `MPCKMeansConstrainedClustering` instance.
    clustering_model = MPCKMeansConstrainedClustering()

    # Run clustering 2 clusters and no constraints.
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
# test_MPCKMeansConstrainedClustering_cluster_with_no_constraints_2
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_with_no_constraints_2():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering works with no `constraints`.
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

    # Initialize a `MPCKMeansConstrainedClustering` instance.
    clustering_model = MPCKMeansConstrainedClustering(eps=0.5, min_samples=3)

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=4,
    )

    assert clustering_model.dict_of_predicted_clusters

    """
    Here, '0' is too far from other points so it is noise
    Furthermore, '7' and '10' are in the same neighbourhood, but no other point.
    They are not numerous enough to create a cluster
    """

    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 3,
        "5": 1,
        "6": 3,
        "7": 2,
        "8": 1,
        "9": 3,
        "10": 2,
        "11": 1,
    }


# ==============================================================================
# test_MPCKMeansConstrainedClustering_cluster_with_some_constraints
# ==============================================================================
def test_MPCKMeansConstrainedClustering_cluster_with_some_constraints():
    """
    Test that the `clustering.mpckmeans.MPCKMeansConstrainedClustering` clustering works with no `constraints`.
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
    clustering_model = MPCKMeansConstrainedClustering()

    # Run clustering 2 clusters and no constraints.
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
