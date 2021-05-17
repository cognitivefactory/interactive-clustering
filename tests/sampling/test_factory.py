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

from cognitivefactory.interactive_clustering.sampling.closest_in_different_clusters import (
    ClosestInDifferentClustersConstraintsSampling,
)
from cognitivefactory.interactive_clustering.sampling.factory import sampling_factory
from cognitivefactory.interactive_clustering.sampling.farhest_in_same_cluster import (
    FarhestInSameClusterConstraintsSampling,
)
from cognitivefactory.interactive_clustering.sampling.random import RandomConstraintsSampling
from cognitivefactory.interactive_clustering.sampling.random_in_same_cluster import (
    RandomInSameClusterConstraintsSampling,
)


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
    Test that the `sampling.sampling_factory` can initialize an instance of `RandomConstraintsSampling`.
    """

    # Check average `random` sampling.
    sampling_model = sampling_factory(
        algorithm="random",
    )
    assert isinstance(sampling_model, RandomConstraintsSampling)


# ==============================================================================
# test_sampling_factory_for_random_in_same_cluster_sampling
# ==============================================================================
def test_sampling_factory_for_random_in_same_cluster_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `RandomInSameClusterConstraintsSampling`.
    """

    # Check average `random_in_same_cluster` sampling.
    sampling_model = sampling_factory(
        algorithm="random_in_same_cluster",
    )
    assert isinstance(sampling_model, RandomInSameClusterConstraintsSampling)


# ==============================================================================
# test_sampling_factory_for_farhest_in_same_cluster_sampling
# ==============================================================================
def test_sampling_factory_for_farhest_in_same_cluster_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `FarhestInSameClusterConstraintsSampling`.
    """

    # Check average `farhest_in_same_cluster` sampling.
    sampling_model = sampling_factory(
        algorithm="farhest_in_same_cluster",
    )
    assert isinstance(sampling_model, FarhestInSameClusterConstraintsSampling)


# ==============================================================================
# test_sampling_factory_for_closest_in_different_clusters_sampling
# ==============================================================================
def test_sampling_factory_for_closest_in_different_clusters_sampling():
    """
    Test that the `sampling.sampling_factory` can initialize an instance of `ClosestInDifferentClustersConstraintsSampling`.
    """

    # Check average `closest_in_different_clusters` sampling.
    sampling_model = sampling_factory(
        algorithm="closest_in_different_clusters",
    )
    assert isinstance(sampling_model, ClosestInDifferentClustersConstraintsSampling)
