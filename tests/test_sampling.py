# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/test_sampling.py
* Description:  Unittests for the `sampling` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# None


# ==============================================================================
# test_sampling_is_importable
# ==============================================================================
def test_sampling_is_importable():
    """
    Test that the `sampling` module is importable.
    """
    from cognitivefactory.interactive_clustering import sampling  # noqa: C0415 (not top level import, it's fine)

    assert sampling
