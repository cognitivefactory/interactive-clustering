# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_spectral.py
* Description:  Unittests for the `clustering.spectral` module.
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

from cognitivefactory.interactive_clustering.clustering.spectral import SpectralConstrainedClustering
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# test_SpectralConstrainedClustering_for_inconsistent_model
# ==============================================================================
def test_SpectralConstrainedClustering_for_inconsistent_model():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` initialization raises an `ValueError` for inconsistent `model` parameter.
    """

    # Check `ValueError` for bad string value for `model`.
    with pytest.raises(ValueError, match="`model`"):
        SpectralConstrainedClustering(
            model="as_you_want",
        )


# ==============================================================================
# test_SpectralConstrainedClustering_for_inconsistent_nb_components
# ==============================================================================
def test_SpectralConstrainedClustering_for_inconsistent_nb_components():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` initialization raises an `ValueError` for inconsistent `nb_components` parameter.
    """

    # Check `ValueError` for bad string value for `nb_components`.
    with pytest.raises(ValueError, match="`nb_components`"):
        SpectralConstrainedClustering(
            nb_components=-1,
        )


# ==============================================================================
# test_SpectralConstrainedClustering_for_correct_settings
# ==============================================================================
def test_SpectralConstrainedClustering_for_correct_settings():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` initialization runs correctly with the correct settings.
    """

    # Check a correct initialization.
    clustering_model = SpectralConstrainedClustering(
        model="SPEC",
        nb_components=100,
        random_seed=3,
    )
    assert clustering_model
    assert clustering_model.model == "SPEC"
    assert clustering_model.nb_components == 100
    assert clustering_model.random_seed == 3


# ==============================================================================
# test_SpectralConstrainedClustering_cluster_for_inconsistent_constraints_manager
# ==============================================================================
def test_SpectralConstrainedClustering_cluster_for_inconsistent_constraints_manager():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` clustering raises an `ValueError` for inconsistent `constraints_manager` parameter.
    """

    # Initialize a `SpectralConstrainedClustering` instance.
    clustering_model = SpectralConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        clustering_model.cluster(
            constraints_manager=None,
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_SpectralConstrainedClustering_cluster_for_inconsistent_vectors
# ==============================================================================
def test_SpectralConstrainedClustering_cluster_for_inconsistent_vectors():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` clustering raises an `ValueError` for inconsistent `vectors` parameter.
    """

    # Initialize a `SpectralConstrainedClustering` instance.
    clustering_model = SpectralConstrainedClustering()

    # Check `ValueError` for not matrix `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors=None,
            nb_clusters=2,
        )


# ==============================================================================
# test_SpectralConstrainedClustering_cluster_for_inconsistent_nb_clusters
# ==============================================================================
def test_SpectralConstrainedClustering_cluster_for_inconsistent_nb_clusters():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` clustering raises an `ValueError` for inconsistent `nb_clusters` parameter.
    """

    # Initialize a `SpectralConstrainedClustering` instance.
    clustering_model = SpectralConstrainedClustering()

    # Check `ValueError` for too small `nb_clusters`.
    with pytest.raises(ValueError, match="`nb_clusters`"):
        clustering_model.cluster(
            constraints_manager=BinaryConstraintsManager(list_of_data_IDs=["first", "second", "third"]),
            vectors={"first": np.array([1, 2, 3]), "second": np.array([[4, 5, 6]]), "third": csr_matrix([7, 8, 9])},
            nb_clusters=-1,
        )


# ==============================================================================
# test_SpectralConstrainedClustering_cluster_model_SPEC_with_no_constraints
# ==============================================================================
def test_SpectralConstrainedClustering_cluster_model_SPEC_with_no_constraints():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` clustering works with SPEC `model` and no constraints.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "0": csr_matrix([1.00, 0.00, 0.00, 0.00]),
        "1": csr_matrix([0.95, 0.02, 0.02, 0.01]),
        "2": csr_matrix([0.98, 0.00, 0.02, 0.00]),
        "3": csr_matrix([0.99, 0.00, 0.01, 0.00]),
        "4": csr_matrix([0.60, 0.17, 0.16, 0.07]),
        "5": csr_matrix([0.60, 0.16, 0.17, 0.07]),
        "6": csr_matrix([0.01, 0.01, 0.01, 0.97]),
        "7": csr_matrix([0.00, 0.01, 0.00, 0.99]),
        "8": csr_matrix([0.00, 0.00, 0.00, 1.00]),
    }
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))

    # Initialize a `SpectralConstrainedClustering` instance.
    clustering_model = SpectralConstrainedClustering(
        model="SPEC",
        random_seed=1,
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
        "6": 2,
        "7": 2,
        "8": 2,
    }


# ==============================================================================
# test_SpectralConstrainedClustering_cluster_model_SPEC_with_some_constraints
# ==============================================================================
def test_SpectralConstrainedClustering_cluster_model_SPEC_with_some_constraints():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` clustering works with SPEC `model` and some constraints.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "0": csr_matrix([1.00, 0.00, 0.00, 0.00]),
        "1": csr_matrix([0.95, 0.02, 0.02, 0.01]),
        "2": csr_matrix([0.98, 0.00, 0.02, 0.00]),
        "3": csr_matrix([0.99, 0.00, 0.01, 0.00]),
        "4": csr_matrix([0.60, 0.17, 0.16, 0.07]),
        "5": csr_matrix([0.60, 0.16, 0.17, 0.07]),
        "6": csr_matrix([0.01, 0.01, 0.01, 0.97]),
        "7": csr_matrix([0.00, 0.01, 0.00, 0.99]),
        "8": csr_matrix([0.00, 0.00, 0.00, 1.00]),
    }
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
    constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="3", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="5", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="7", data_ID2="8", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="2", data_ID2="4", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="4", data_ID2="7", constraint_type="CANNOT_LINK")

    # Initialize a `SpectralConstrainedClustering` instance.
    clustering_model = SpectralConstrainedClustering(
        model="SPEC",
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
        "6": 2,
        "7": 2,
        "8": 2,
    }


# ==============================================================================
# test_SpectralConstrainedClustering_cluster_model_SPEC_with_full_constraints
# ==============================================================================
def test_SpectralConstrainedClustering_cluster_model_SPEC_with_full_constraints():
    """
    Test that the `clustering.spectral.SpectralConstrainedClustering` clustering works with SPEC `model` and full constraints.
    """

    # Define `vectors` and `constraints_manager`
    vectors = {
        "0": csr_matrix([1.00, 0.00, 0.00, 0.00]),
        "1": csr_matrix([0.95, 0.02, 0.02, 0.01]),
        "2": csr_matrix([0.98, 0.00, 0.02, 0.00]),
        "3": csr_matrix([0.99, 0.00, 0.01, 0.00]),
        "4": csr_matrix([0.60, 0.17, 0.16, 0.07]),
        "5": csr_matrix([0.60, 0.16, 0.17, 0.07]),
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

    # Initialize a `SpectralConstrainedClustering` instance.
    clustering_model = SpectralConstrainedClustering(
        model="SPEC",
        random_seed=1,
    )

    # Run clustering 3 clusters and full constraints.
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
