# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/test_clustering.py
* Description:  Unittests for the `clustering` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# None


# ==============================================================================
# test_clustering_is_importable
# ==============================================================================
def test_clustering_is_importable():
    """
    Test that the `clustering` module is importable.
    """
    from cognitivefactory.interactive_clustering import clustering  # noqa: C0415 (not top level import, it's fine)

    assert clustering
