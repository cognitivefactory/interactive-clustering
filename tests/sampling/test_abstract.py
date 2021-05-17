# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/sampling/test_abstract.py
* Description:  Unittests for the `sampling.abstract` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.sampling.abstract import AbstractConstraintsSampling


# ==============================================================================
# test_AbstractConstraintsSampling_is_abstract
# ==============================================================================
def test_AbstractConstraintsSampling_is_abstract():
    """
    Test that the `sampling.abstract.AbstractConstraintsSampling` class is abstract.
    """

    # Check `TypeError` for initialization of `AbstractConstraintsSampling`.
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractConstraintsSampling()
