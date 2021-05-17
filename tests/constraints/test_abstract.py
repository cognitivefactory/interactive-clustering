# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/constraints/test_abstract.py
* Description:  Unittests for the `constraints.abstract` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager


# ==============================================================================
# test_AbstractConstraintsManager_is_abstract
# ==============================================================================
def test_AbstractConstraintsManager_is_abstract():
    """
    Test that the `constraints.abstract.AbstractConstraintsManager` class is abstract.
    """

    # Check `TypeError` for initialization of `AbstractConstraintsManager`.
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractConstraintsManager()
