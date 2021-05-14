# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/test_constraints.py
* Description:  Unittests for the `constraints` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# None


# ==============================================================================
# test_constraints_is_importable
# ==============================================================================
def test_constraints_is_importable():
    """
    Test that the `constraints` module is importable.
    """
    from cognitivefactory.interactive_clustering import constraints  # noqa: C0415 (not top level import, it's fine)

    assert constraints
