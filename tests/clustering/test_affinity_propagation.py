"""
* Name:         interactive-clustering/tests/clustering/test_affinity_propagation.py
* Description:  Unittests for the `clustering.affinity_propagation` module.
* Author:       David NICOLAZO, Esther LENOTRE, Marc TRUTT
* Created:      02/11/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================


import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.clustering.affinity_propagation import (
    AffinityPropagationConstrainedClustering,
)
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_for_inconsistent_max_iteration
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_for_inconsistent_max_iteration():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` initialization raises an `ValueError` for inconsistent `max_iteration` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`max_iteration`"):
        AffinityPropagationConstrainedClustering(
            max_iteration=-1,
        )


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_for_inconsistent_convergence_iteration
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_for_inconsistent_convergence_iteration():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` initialization raises an `ValueError` for inconsistent `convergence_iteration` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`convergence_iteration`"):
        AffinityPropagationConstrainedClustering(
            convergence_iteration=-1,
        )


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_for_correct_settings
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_for_correct_settings():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` initialization runs correctly with the correct settings.
    """

    # Check a correct initialization.
    clustering_model = AffinityPropagationConstrainedClustering(
        max_iteration=100,
        convergence_iteration=5,
    )
    assert clustering_model
    assert clustering_model.max_iteration == 100
    assert clustering_model.convergence_iteration == 5


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_cluster_for_inconsistent_constraints_manager
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_for_inconsistent_constraints_manager():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints_manager` parameter.
    """

    # Initialize a `AffinityPropagationConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            constraints_manager=None,
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `AffinityPropagationConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_cluster_for_inconsistent_nb_clusters
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_for_inconsistent_nb_clusters_1():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `AffinityPropagationConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=2,
        )


# ==============================================================================
# test_AffinityPropagationConstrainedClustering_cluster_with_no_constraints_1
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_with_no_constraints_1():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` clustering works with no `constraints`.
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

    # Initialize a `AffinityPropagationConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

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
# test_AffinityPropagationConstrainedClustering_cluster_with_no_constraints_2
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_with_no_constraints_2():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` clustering works with no `constraints`.
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

    # Initialize a `AffinityPropagationConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Run clustering with no constraints.
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
# test_AffinityPropagationConstrainedClustering_cluster_with_some_constraints
# ==============================================================================
def test_AffinityPropagationConstrainedClustering_cluster_with_some_constraints():
    """
    Test that the `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` clustering works with some `constraints`.
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

    # Initialize a `AffinityPropagationConstrainedClustering` instance.
    clustering_model = AffinityPropagationConstrainedClustering()

    # Run clustering with some constraints.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
    )

    assert clustering_model.dict_of_predicted_clusters
    assert dict_of_predicted_clusters == {
        "0": 0,
        "1": 1,
        "2": 0,  # TODO: 2,
        "3": 0,  # TODO: 2,
        "4": 2,  # TODO: 3,
        "5": 0,  # TODO: 2,
        "6": 2,  # TODO: 3,
        "7": 0,
        "8": 1,
        "9": 2,  # TODO: 3,
        "10": 0,
        "11": 1,
    }
