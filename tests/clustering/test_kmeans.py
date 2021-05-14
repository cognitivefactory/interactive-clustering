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

# Dependencies needed to handle matrix.
import numpy as np

# Needed library to apply tests.
import pytest
from scipy.sparse import csr_matrix

# Modules/Classes/Methods to test.
from cognitivefactory.interactive_clustering.clustering.kmeans import KMeansConstrainedClustering

# Dependency needed to manage constraints.
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
# test_KMeansConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            vectors=None,
            nb_clusters=2,
        )

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            vectors="this_is_my_vectors",
            nb_clusters=2,
        )

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`list_of_data_IDs`"):
        clustering_model.cluster(
            vectors={1: "yolo", 2: "yolo 2"},
            nb_clusters=2,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_for_inconsistent_nb_clusters():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=-1,
        )


# ==============================================================================
# test_KMeansConstrainedClustering_cluster_for_inconsistent_constraints
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_for_inconsistent_constraints():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints` parameter.
    """

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering()

    # Check `ValueError` for not dictionary `constraints`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=2,
            constraints_manager="this_is_my_constraints",
        )

    # Check `ValueError` for not well defined `constraints`.

    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=2,
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third", "fourth"]),
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
    constraints_manager = None

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=2,
    )

    # Run clustering 2 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        vectors=vectors,
        nb_clusters=2,
        constraints_manager=constraints_manager,
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
    constraints_manager = None

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=2,
    )

    # Run clustering 3 clusters and no constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        vectors=vectors,
        nb_clusters=3,
        constraints_manager=constraints_manager,
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
        vectors=vectors,
        nb_clusters=3,
        constraints_manager=constraints_manager,
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
        vectors=vectors,
        nb_clusters=4,
        constraints_manager=constraints_manager,
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
# test_KMeansConstrainedClustering_cluster_with_verbose_output_1
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_verbose_output_1():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with option `verbose`.
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

    # Run clustering with verbose output.
    dict_of_predicted_clusters = clustering_model.cluster(
        vectors=vectors,
        nb_clusters=5,
        constraints_manager=constraints_manager,
        verbose=True,
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
# test_KMeansConstrainedClustering_cluster_with_verbose_output_2
# ==============================================================================
def test_KMeansConstrainedClustering_cluster_with_verbose_output_2():
    """
    Test that the `clustering.kmeans.KMeansConstrainedClustering` clustering works with option `verbose`.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "00": csr_matrix([1.00, 0.00, 0.45]),
        "01": csr_matrix([0.80, 0.99, 0.01]),
        "02": csr_matrix([1.00, 0.00, 0.15]),
        "03": csr_matrix([1.00, 0.86, 0.00]),
        "04": csr_matrix([0.95, 0.28, 0.01]),
        "05": csr_matrix([0.81, 0.02, 0.01]),
        "06": csr_matrix([0.95, 0.04, 0.01]),
        "07": csr_matrix([0.95, 0.02, 0.05]),
        "08": csr_matrix([0.80, 1.00, 0.52]),
        "09": csr_matrix([0.98, 0.10, 0.54]),
        "10": csr_matrix([0.98, 0.15, 0.10]),
        "11": csr_matrix([0.90, 0.00, 0.00]),
        "12": csr_matrix([0.99, 0.01, 0.28]),
        "13": csr_matrix([0.50, 0.22, 0.07]),
        "14": csr_matrix([0.49, 0.21, 0.15]),
        "15": csr_matrix([0.49, 0.35, 0.35]),
        "16": csr_matrix([0.49, 0.20, 0.36]),
        "17": csr_matrix([0.50, 0.40, 0.75]),
        "18": csr_matrix([0.99, 0.28, 0.99]),
        "19": csr_matrix([0.99, 0.28, 1.01]),
        "20": csr_matrix([1.00, 0.99, 1.00]),
        "21": csr_matrix([0.70, 0.28, 0.85]),
        "22": csr_matrix([0.99, 0.28, 0.99]),
        "23": csr_matrix([0.99, 0.35, 0.99]),
        "24": csr_matrix([0.51, 0.28, 0.12]),
        "25": csr_matrix([0.99, 0.01, 0.97]),
        "26": csr_matrix([0.01, 0.10, 0.97]),
        "27": csr_matrix([0.15, 0.86, 0.99]),
        "28": csr_matrix([0.80, 0.01, 0.99]),
        "29": csr_matrix([0.25, 0.02, 0.98]),
        "30": csr_matrix([0.37, 0.99, 0.98]),
        "31": csr_matrix([0.22, 0.01, 0.99]),
        "32": csr_matrix([0.45, 0.00, 1.00]),
        "33": csr_matrix([1.00, 0.00, 0.00]),
        "34": csr_matrix([1.00, 0.89, 0.49]),
        "35": csr_matrix([0.35, 0.51, 0.00]),
        "36": csr_matrix([0.35, 0.52, 0.00]),
        "37": csr_matrix([0.96, 0.72, 0.03]),
        "38": csr_matrix([0.25, 0.58, 0.00]),
        "39": csr_matrix([0.54, 0.28, 0.00]),
        "40": csr_matrix([0.40, 0.37, 0.00]),
        "41": csr_matrix([1.00, 1.00, 0.64]),
        "42": csr_matrix([1.00, 0.73, 1.00]),
        "43": csr_matrix([0.55, 0.00, 0.00]),
        "44": csr_matrix([1.00, 0.75, 0.22]),
        "45": csr_matrix([0.56, 0.83, 0.33]),
        "46": csr_matrix([0.36, 0.65, 0.63]),
        "47": csr_matrix([1.00, 0.88, 0.00]),
        "48": csr_matrix([1.00, 0.00, 0.00]),
        "49": csr_matrix([0.00, 1.00, 0.66]),
        "50": csr_matrix([1.00, 0.34, 0.15]),
        "51": csr_matrix([1.00, 1.00, 0.51]),
        "52": csr_matrix([0.78, 0.04, 1.00]),
        "53": csr_matrix([0.76, 1.00, 0.95]),
        "54": csr_matrix([0.12, 1.00, 1.00]),
        "55": csr_matrix([1.00, 0.46, 1.00]),
        "56": csr_matrix([0.20, 0.00, 1.00]),
        "57": csr_matrix([0.82, 0.74, 0.08]),
        "58": csr_matrix([0.44, 0.00, 0.61]),
        "59": csr_matrix([0.10, 0.00, 1.00]),
        "60": csr_matrix([0.99, 1.00, 0.12]),
        "61": csr_matrix([1.00, 0.75, 0.99]),
        "62": csr_matrix([1.00, 0.65, 0.87]),
        "63": csr_matrix([0.35, 1.00, 0.40]),
        "64": csr_matrix([0.61, 0.87, 0.09]),
        "65": csr_matrix([0.79, 0.00, 0.65]),
    }
    constraints_manager = None

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        random_seed=3,
    )

    # Run clustering with verbose output.
    dict_of_predicted_clusters = clustering_model.cluster(
        vectors=vectors,
        nb_clusters=12,
        constraints_manager=constraints_manager,
        verbose=True,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters


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
    constraints_manager = None

    # Initialize a `KMeansConstrainedClustering` instance.
    clustering_model = KMeansConstrainedClustering(
        max_iteration=1,
    )

    # Run clustering with verbose output.
    dict_of_predicted_clusters = clustering_model.cluster(
        vectors=vectors,
        nb_clusters=2,
        constraints_manager=constraints_manager,
        verbose=True,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters
