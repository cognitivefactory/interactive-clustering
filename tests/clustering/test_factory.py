# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_factory.py
* Description:  Unittests for the `clustering.factory` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import math

import pytest

from cognitivefactory.interactive_clustering.clustering.factory import clustering_factory
from cognitivefactory.interactive_clustering.clustering.hierarchical import HierarchicalConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.kmeans import KMeansConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.spectral import SpectralConstrainedClustering


# ==============================================================================
# test_clustering_factory_for_not_implemented_clustering
# ==============================================================================
def test_clustering_factory_for_not_implemented_clustering():
    """
    Test that the `clustering.factory.clustering_factory` method raises an `ValueError` for not implemented clustering.
    """

    # Check `ValueError` for bad string value for `algorithm`.
    with pytest.raises(ValueError, match="`algorithm`"):
        clustering_factory(
            algorithm="unknown",
        )


# ==============================================================================
# test_clustering_factory_for_hierarchical_clustering
# ==============================================================================
def test_clustering_factory_for_hierarchical_clustering():
    """
    Test that the `clustering.factory.clustering_factory` can initialize an instance of `HierarchicalConstrainedClustering`.
    """

    # Check average `hierarchical` clustering.
    clustering_model = clustering_factory(
        algorithm="hierarchical",
        linkage="average",
    )
    assert isinstance(clustering_model, HierarchicalConstrainedClustering)
    assert clustering_model.linkage == "average"

    # Check single `hierarchical` clustering.
    clustering_model = clustering_factory(
        algorithm="hierarchical",
        linkage="single",
    )
    assert isinstance(clustering_model, HierarchicalConstrainedClustering)
    assert clustering_model.linkage == "single"


# ==============================================================================
# test_clustering_factory_for_kmeans_clustering
# ==============================================================================
def test_clustering_factory_for_kmeans_clustering():
    """
    Test that the `clustering.factory.clustering_factory` can initialize an instance of `KMeansConstrainedClustering`.
    """

    # Check COP `kmeans` clustering.
    clustering_model = clustering_factory(
        algorithm="kmeans",
        model="COP",
        max_iteration=100,
        tolerance=1e-3,
    )
    assert isinstance(clustering_model, KMeansConstrainedClustering)
    assert clustering_model.model == "COP"
    assert clustering_model.max_iteration == 100
    assert math.isclose(clustering_model.tolerance, 1e-3)


# ==============================================================================
# test_clustering_factory_for_spectral_clustering
# ==============================================================================
def test_clustering_factory_for_spectral_clustering():
    """
    Test that the `clustering.factory.clustering_factory` can initialize an instance of `SpectralConstrainedClustering`.
    """

    # Check SPEC `spectral` clustering.
    clustering_model = clustering_factory(
        algorithm="spectral",
        model="SPEC",
        nb_components=10,
    )
    assert isinstance(clustering_model, SpectralConstrainedClustering)
    assert clustering_model.model == "SPEC"
    assert clustering_model.nb_components == 10

    """ #TODO add when corrected.
    # Check CCSR `spectral` clustering.
    clustering_model = clustering_factory(
        algorithm="spectral",
        model="CCSR",
        nb_components=10,
    )
    assert isinstance(clustering_model, SpectralConstrainedClustering)
    assert clustering_model.model == "CCSR"
    assert clustering_model.nb_components == 10
    """
