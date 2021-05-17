# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/constraints/test_factory.py
* Description:  Unittests for the `constraints.factory` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
from cognitivefactory.interactive_clustering.constraints.factory import managing_factory


# ==============================================================================
# test_managing_factory_for_not_implemented_constraints_manager
# ==============================================================================
def test_managing_factory_for_not_implemented_constraints_manager():
    """
    Test that the `constraints.factory.managing_factory` method raises an `ValueError` for not implemented constraints manager.
    """

    # Check `ValueError` for bad string value for `manager`.
    with pytest.raises(ValueError, match="`manager`"):
        managing_factory(manager="unknown", list_of_data_IDs=["first", "second", "third"])


# ==============================================================================
# test_managing_factory_for_binary_constraints_manager
# ==============================================================================
def test_managing_factory_for_binary_constraints_manager():
    """
    Test that the `constraints.factory.managing_factory` can initialize an instance of `BinaryConstraintsManager`.
    """

    # Check `hierarchical` clustering.
    constraints_manager = managing_factory(algorithm="binary", list_of_data_IDs=["first", "second", "third"])
    assert isinstance(constraints_manager, BinaryConstraintsManager)
