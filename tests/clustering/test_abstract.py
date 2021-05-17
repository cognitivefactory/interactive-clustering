# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/clustering/test_abstract.py
* Description:  Unittests for the `clustering.abstract` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.clustering.abstract import (
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)


# ==============================================================================
# test_AbstractConstrainedClustering_is_abstract
# ==============================================================================
def test_AbstractConstrainedClustering_is_abstract():
    """
    Test that the `clustering.abstract.AbstractConstrainedClustering` class is abstract.
    """

    # Check `TypeError` for initialization of `AbstractConstrainedClustering`.
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractConstrainedClustering()


# ==============================================================================
# test_rename_clusters_by_order
# ==============================================================================
def test_rename_clusters_by_order():
    """
    Test that the `clustering.abstract.test_rename_clusters_by_order` mathod works.
    """

    assert rename_clusters_by_order(clusters={"d": 2, "b": 0, "a": 1, "e": 0, "c": 1}) == {
        "a": 0,
        "b": 1,
        "c": 0,
        "d": 2,
        "e": 1,
    }
