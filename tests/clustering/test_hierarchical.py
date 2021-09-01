# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_hierarchical.py
* Description:  Unittests for the `clustering.hierarchical` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.clustering.hierarchical import Cluster, HierarchicalConstrainedClustering
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# test_HierarchicalConstrainedClustering_for_inconsistent_linkage
# ==============================================================================
def test_HierarchicalConstrainedClustering_for_inconsistent_linkage():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` initialization raises an `ValueError` for inconsistent `linkage` parameter.
    """

    # Check `ValueError` for bad string value for `linkage`.
    with pytest.raises(ValueError, match="`linkage`"):
        HierarchicalConstrainedClustering(
            linkage="as_you_want",
        )


# ==============================================================================
# test_HierarchicalConstrainedClustering_for_correct_settings
# ==============================================================================
def test_HierarchicalConstrainedClustering_for_correct_settings():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` initialization runs correctly with the correct settings.
    """

    # Check a correct initialization.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="average",
        random_seed=2,
    )
    assert clustering_model
    assert clustering_model.linkage == "average"
    assert clustering_model.random_seed == 2


# ==============================================================================
# test_HierarchicalConstrainedClustering_cluster_for_inconsistent_constraints_manager
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_for_inconsistent_constraints_manager():
    """
    Test that the `clustering.spectral.HierarchicalConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints_manager` parameter.
    """

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            constraints_manager=None,
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_HierarchicalConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.spectral.HierarchicalConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_HierarchicalConstrainedClustering_cluster_for_inconsistent_nb_clusters
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_for_inconsistent_nb_clusters():
    """
    Test that the `clustering.spectral.HierarchicalConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=-1,
        )


# ==============================================================================
# test_HierarchicalConstrainedClustering_cluster_with_ward_linkage
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_ward_linkage():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with ward `linkage`.
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="7", constraint_type="CANNOT_LINK")

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="ward",
        random_seed=1,
    )

    # Run clustering 3 clusters and some constraints.
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
# test_HierarchicalConstrainedClustering_cluster_with_average_linkage
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_average_linkage():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with average `linkage`.
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="7", constraint_type="CANNOT_LINK")

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="average",
        random_seed=1,
    )

    # Run clustering 3 clusters and some constraints.
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
# test_HierarchicalConstrainedClustering_cluster_with_single_linkage
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_single_linkage():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with single `linkage`.
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="7", constraint_type="CANNOT_LINK")

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="single",
        random_seed=1,
    )

    # Run clustering 3 clusters and some constraints.
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
# test_HierarchicalConstrainedClustering_cluster_with_complete_linkage
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_complete_linkage():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with complete `linkage`.
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
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="5", data_ID2="7", constraint_type="CANNOT_LINK")

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="complete",
        random_seed=1,
    )

    # Run clustering 3 clusters and some constraints.
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
# test_HierarchicalConstrainedClustering_cluster_with_no_constraints_1
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_no_constraints_1():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with no `constraints`.
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

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
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
# test_HierarchicalConstrainedClustering_cluster_with_no_constraints_2
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_no_constraints_2():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with no `constraints`.
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

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
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
# test_HierarchicalConstrainedClustering_cluster_with_some_constraints
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_some_constraints():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with some `constraints`.
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
    constraints_manager.add_constraint(data_ID1="0", data_ID2="6", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="7", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="8", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="5", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="4", constraint_type="CANNOT_LINK")

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
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
# test_HierarchicalConstrainedClustering_cluster_with_full_constraints
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_full_constraints():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with full `constraints`.
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

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering()

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
# test_HierarchicalConstrainedClustering_compute_predicted_clusters_without_clustering_tree
# ==============================================================================
def test_HierarchicalConstrainedClustering_compute_predicted_clusters_without_clustering_tree():
    """
    Test that the `compute_predicted_clusters` method of the `HierarchicalConstrainedClustering` raises `ValueError` if clustering is not run.
    """

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="single",
        random_seed=1,
    )

    # Run `compute_predicted_clusters` without computing the clustering tree.
    with pytest.raises(ValueError, match="`clustering_root`"):
        clustering_model.compute_predicted_clusters(
            nb_clusters=2,
            by="size",
        )


# ==============================================================================
# test_HierarchicalConstrainedClustering_compute_predicted_clusters_travelling_by_size
# ==============================================================================
def test_HierarchicalConstrainedClustering_compute_predicted_clusters_travelling_by_size():
    """
    Test that the `compute_predicted_clusters` method of the `HierarchicalConstrainedClustering` clustering works by travelling by `"size"`.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "00": csr_matrix([1.00, 0.00, 0.00]),
        "01": csr_matrix([0.99, 0.00, 0.00]),
        "02": csr_matrix([0.97, 0.00, 0.00]),
        "03": csr_matrix([0.96, 0.00, 0.00]),
        "04": csr_matrix([0.94, 0.00, 0.00]),
        "05": csr_matrix([0.93, 0.00, 0.00]),
        "06": csr_matrix([0.80, 0.80, 0.00]),
        "07": csr_matrix([0.80, 0.81, 0.00]),
        "08": csr_matrix([0.00, 0.00, 0.70]),
        "09": csr_matrix([0.00, 0.00, 0.71]),
        "10": csr_matrix([0.00, 0.00, 0.99]),
        "11": csr_matrix([0.00, 0.00, 1.00]),
    }
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="single",
        random_seed=1,
    )

    # Compute all clustering tree.
    clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=2,
    )

    # Run `compute_predicted_clusters` while travalleing clustering tree by `"size"`.
    dict_of_predicted_clusters = clustering_model.compute_predicted_clusters(
        nb_clusters=4,
        by="size",
    )
    assert dict_of_predicted_clusters == {
        "00": 0,
        "01": 0,
        "02": 0,
        "03": 0,
        "04": 1,
        "05": 1,
        "06": 2,
        "07": 2,
        "08": 3,
        "09": 3,
        "10": 3,
        "11": 3,
    }


# ==============================================================================
# test_HierarchicalConstrainedClustering_compute_predicted_clusters_travelling_by_iteration
# ==============================================================================
def test_HierarchicalConstrainedClustering_compute_predicted_clusters_travelling_by_iteration():
    """
    Test that the `compute_predicted_clusters` method of the `HierarchicalConstrainedClustering` clustering works by travelling by `"iteration"`.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "00": csr_matrix([1.00, 0.00, 0.00]),
        "01": csr_matrix([0.99, 0.00, 0.00]),
        "02": csr_matrix([0.97, 0.00, 0.00]),
        "03": csr_matrix([0.96, 0.00, 0.00]),
        "04": csr_matrix([0.94, 0.00, 0.00]),
        "05": csr_matrix([0.93, 0.00, 0.00]),
        "06": csr_matrix([0.80, 0.80, 0.00]),
        "07": csr_matrix([0.80, 0.81, 0.00]),
        "08": csr_matrix([0.00, 0.00, 0.70]),
        "09": csr_matrix([0.00, 0.00, 0.71]),
        "10": csr_matrix([0.00, 0.00, 0.99]),
        "11": csr_matrix([0.00, 0.00, 1.00]),
    }
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="single",
        random_seed=1,
    )

    # Compute all clustering tree.
    clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=2,
    )

    # Run `compute_predicted_clusters` while travalleing clustering tree by `"iteration"`.
    dict_of_predicted_clusters = clustering_model.compute_predicted_clusters(
        nb_clusters=4,
        by="iteration",
    )
    assert dict_of_predicted_clusters == {
        "00": 0,
        "01": 0,
        "02": 0,
        "03": 0,
        "04": 0,
        "05": 0,
        "06": 1,
        "07": 1,
        "08": 2,
        "09": 2,
        "10": 3,
        "11": 3,
    }


# ==============================================================================
# test_HierarchicalConstrainedClustering_cluster_with_break_loop
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_with_break_loop():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering can break clustering loop.
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

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="average",
    )

    # Run clustering.
    clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=5,
    )
    assert clustering_model.dict_of_predicted_clusters
    assert clustering_model.dict_of_predicted_clusters == {
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
# test_HierarchicalConstrainedClustering_cluster_end_cases_with_too_many_clusters
# ==============================================================================
def test_HierarchicalConstrainedClustering_cluster_end_cases_with_too_many_clusters():
    """
    Test that the `clustering.hierarchical.HierarchicalConstrainedClustering` clustering works with too many clusters.
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

    # Initialize a `HierarchicalConstrainedClustering` instance.
    clustering_model = HierarchicalConstrainedClustering(
        linkage="average",
    )

    # Run clustering.
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=vectors,
        nb_clusters=99,
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
# test_Cluster_for_inconsistent_children_and_members
# ==============================================================================
def test_Cluster_for_inconsistent_children_and_members():
    """
    Test that the `clustering.hierarchical.Cluster` initialization raises an `ValueError` for inconsistent `children` and `members` parameters.
    """

    # Define `vectors`.
    vectors = {
        "0": csr_matrix([1.00, 0.00]),
        "1": csr_matrix([0.99, 0.01]),
        "2": csr_matrix([0.02, 0.98]),
        "3": csr_matrix([0.01, 0.99]),
        "4": csr_matrix([0.00, 1.00]),
    }

    # Check `ValueError` for both `children` and `members` unset.
    with pytest.raises(ValueError, match="by `children` setting or by `members` setting"):
        Cluster(vectors=vectors, cluster_ID=2, clustering_iteration=1, children=None, members=None)

    # Check `ValueError` for both `children` and `members` set.
    with pytest.raises(ValueError, match="by `children` setting or by `members` setting"):
        Cluster(
            vectors=vectors,
            cluster_ID=2,
            clustering_iteration=1,
            children=[
                Cluster(
                    vectors=vectors,
                    cluster_ID=0,
                    clustering_iteration=0,
                    members=["0", "1"],
                ),
                Cluster(
                    vectors=vectors,
                    cluster_ID=1,
                    clustering_iteration=0,
                    members=["2", "3", "4"],
                ),
            ],
            members=["5", "6", "7", "8", "9"],
        )


# ==============================================================================
# test_Cluster_to_dict
# ==============================================================================
def test_Cluster_add_new_children():
    """
    Test that the `clustering.hierarchical.Cluster.add_new_children` method of `Cluster` class works.
    """

    # Define `vectors`.
    vectors = {
        "0": csr_matrix([1.00, 0.00]),
        "1": csr_matrix([0.99, 0.01]),
        "2": csr_matrix([0.02, 0.98]),
        "3": csr_matrix([0.01, 0.99]),
        "4": csr_matrix([0.00, 1.00]),
    }

    # Create `clusters`.
    clusters = Cluster(
        vectors=vectors,
        cluster_ID=2,
        clustering_iteration=1,
        children=[
            Cluster(
                vectors=vectors,
                cluster_ID=0,
                clustering_iteration=0,
                members=["0", "1"],
            ),
        ],
    )

    assert clusters.members == ["0", "1"]
    assert clusters.clustering_iteration == 1
    assert clusters.get_cluster_size() == 2

    clusters.add_new_children(
        new_children=[
            Cluster(
                vectors=vectors,
                cluster_ID=1,
                clustering_iteration=0,
                members=["2", "3", "4"],
            ),
        ],
        new_clustering_iteration=2,
    )

    assert clusters.members == ["0", "1", "2", "3", "4"]
    assert clusters.clustering_iteration == 2
    assert clusters.get_cluster_size() == 5


# ==============================================================================
# test_Cluster_to_dict
# ==============================================================================
def test_Cluster_to_dict():
    """
    Test that the `clustering.hierarchical.Cluster.to_dict` method of `Cluster` class works.
    """

    # Define `vectors`.
    vectors = {
        "0": csr_matrix([1.00, 0.00]),
        "1": csr_matrix([0.99, 0.01]),
        "2": csr_matrix([0.02, 0.98]),
        "3": csr_matrix([0.01, 0.99]),
        "4": csr_matrix([0.00, 1.00]),
    }

    # Create `clusters`.
    clusters = Cluster(
        vectors=vectors,
        cluster_ID=2,
        clustering_iteration=1,
        children=[
            Cluster(
                vectors=vectors,
                cluster_ID=0,
                clustering_iteration=0,
                members=["0", "1"],
            ),
            Cluster(
                vectors=vectors,
                cluster_ID=1,
                clustering_iteration=0,
                members=["2", "3", "4"],
            ),
        ],
    )

    # Define expected dictionnary.
    dict_expected = {
        "cluster_ID": 2,
        "clustering_iteration": 1,
        "children": [
            {
                "cluster_ID": 0,
                "clustering_iteration": 0,
                "children": [],
                "cluster_inverse_depth": 0,
                "members": ["0", "1"],
            },
            {
                "cluster_ID": 1,
                "clustering_iteration": 0,
                "children": [],
                "cluster_inverse_depth": 0,
                "members": ["2", "3", "4"],
            },
        ],
        "cluster_inverse_depth": 1,
        "members": ["0", "1", "2", "3", "4"],
    }

    assert clusters.to_dict() == dict_expected
