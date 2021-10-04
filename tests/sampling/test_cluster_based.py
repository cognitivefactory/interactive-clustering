# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/sampling/test_cluster_based.py
* Description:  Unittests for the `sampling.cluster_based` module.
* Author:       Erwan Schild
* Created:      04/10/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.sampling.clusters_based import ClustersBasedConstraintsSampling


# ==============================================================================
# test_cluster_based_for_incorrect_clusters_restriction_parameter
# ==============================================================================
def test_cluster_based_for_incorrect_clusters_restriction_parameter():
    """
    Test that the `sampling.cluster_based.ClustersBasedConstraintsSampling` initialization raises an `ValueError` for incorrect `clusters_restriction` parameter.
    """

    # Check `ValueError` for bad string value for `clusters_restriction`.
    with pytest.raises(ValueError, match="`clusters_restriction`"):
        ClustersBasedConstraintsSampling(
            clusters_restriction="unknown",
        )


# ==============================================================================
# test_cluster_based_for_incorrect_distance_restriction_parameter
# ==============================================================================
def test_cluster_based_for_incorrect_distance_restriction_parameter():
    """
    Test that the `sampling.cluster_based.ClustersBasedConstraintsSampling` initialization raises an `ValueError` for incorrect `distance_restriction` parameter.
    """

    # Check `ValueError` for bad string value for `distance_restriction`.
    with pytest.raises(ValueError, match="`distance_restriction`"):
        ClustersBasedConstraintsSampling(
            distance_restriction="unknown",
        )


# ==============================================================================
# test_cluster_based_for_incorrect_without_added_constraints_parameter
# ==============================================================================
def test_cluster_based_for_incorrect_without_added_constraints_parameter():
    """
    Test that the `sampling.cluster_based.ClustersBasedConstraintsSampling` initialization raises an `ValueError` for incorrect `without_added_constraints` parameter.
    """

    # Check `ValueError` for bad string value for `without_added_constraints`.
    with pytest.raises(ValueError, match="`without_added_constraints`"):
        ClustersBasedConstraintsSampling(
            without_added_constraints="unknown",
        )


# ==============================================================================
# test_cluster_based_for_incorrect_without_inferred_constraints_parameter
# ==============================================================================
def test_cluster_based_for_incorrect_without_inferred_constraints_parameter():
    """
    Test that the `sampling.cluster_based.ClustersBasedConstraintsSampling` initialization raises an `ValueError` for incorrect `without_inferred_constraints` parameter.
    """

    # Check `ValueError` for bad string value for `without_inferred_constraints`.
    with pytest.raises(ValueError, match="`without_inferred_constraints`"):
        ClustersBasedConstraintsSampling(
            without_inferred_constraints="unknown",
        )
