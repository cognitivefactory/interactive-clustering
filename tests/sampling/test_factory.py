# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/sampling/test_factory.py
* Description:  Unittests for the `sampling.factory` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.sampling.clusters_based import ClustersBasedConstraintsSampling
from cognitivefactory.interactive_clustering.sampling.factory import sampling_factory


# ==============================================================================
# test_sampling_factory_for_not_implemented_clustering
# ==============================================================================
def test_sampling_factory_for_not_implemented_clustering():
    """
    Test that the `sampling.sampling_factory` method raises an `ValueError` for not implemented clustering.
    """

    # Check `ValueError` for bad string value for `algorithm`.
    with pytest.raises(ValueError, match="`algorithm`"):
        sampling_factory(
            algorithm="unknown",
        )


# ==============================================================================
# test_sampling_factory_for_random_sampling
# ==============================================================================
def test_sampling_factory_for_random_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `ClustersBasedConstraintsSampling` to sample random pairs of data IDs.
    """

    # Check average `random` sampling.
    sampling_model = sampling_factory(
        algorithm="random",
    )
    assert isinstance(sampling_model, ClustersBasedConstraintsSampling)
    assert sampling_model.clusters_restriction is None
    assert sampling_model.distance_restriction is None


# ==============================================================================
# test_sampling_factory_for_random_in_same_cluster_sampling
# ==============================================================================
def test_sampling_factory_for_random_in_same_cluster_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `ClustersBasedConstraintsSampling` to sample random pairs of data IDs in same clusters.
    """

    # Check average `random_in_same_cluster` sampling.
    sampling_model = sampling_factory(
        algorithm="random_in_same_cluster",
    )
    assert isinstance(sampling_model, ClustersBasedConstraintsSampling)
    assert sampling_model.clusters_restriction == "same_cluster"
    assert sampling_model.distance_restriction is None


# ==============================================================================
# test_sampling_factory_for_farthest_in_same_cluster_sampling
# ==============================================================================
def test_sampling_factory_for_farthest_in_same_cluster_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `ClustersBasedConstraintsSampling` to sample farthest pairs of data IDs in same clusters.
    """

    # Check average `farthest_in_same_cluster` sampling.
    sampling_model = sampling_factory(
        algorithm="farthest_in_same_cluster",
    )
    assert isinstance(sampling_model, ClustersBasedConstraintsSampling)
    assert sampling_model.clusters_restriction == "same_cluster"
    assert sampling_model.distance_restriction == "farthest_neighbors"


# ==============================================================================
# test_sampling_factory_for_closest_in_different_clusters_sampling
# ==============================================================================
def test_sampling_factory_for_closest_in_different_clusters_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `ClustersBasedConstraintsSampling` to sample closest pairs of data IDs in different clusters.
    """

    # Check average `closest_in_different_clusters` sampling.
    sampling_model = sampling_factory(
        algorithm="closest_in_different_clusters",
    )
    assert isinstance(sampling_model, ClustersBasedConstraintsSampling)
    assert sampling_model.clusters_restriction == "different_clusters"
    assert sampling_model.distance_restriction == "closest_neighbors"
