# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/tests_utils.py
* Description:  Unittests for the `utils` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# None


# ==============================================================================
# test_utils_is_importable :
# ==============================================================================
def test_utils_is_importable():
    """
    Test that the `utils` module is importable.
    """
    from cognitivefactory.interactive_clustering import utils  # noqa: C0415 (not top level import, it"s fine)

    assert utils
