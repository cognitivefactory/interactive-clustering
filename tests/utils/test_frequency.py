# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/utils/test_frequency.py
* Description:  Unittests for the `utils.frequency` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from cognitivefactory.interactive_clustering.utils.frequency import compute_clusters_frequency


# ==============================================================================
# test_compute_clusters_frequency_for_no_clustering_result
# ==============================================================================
def test_compute_clusters_frequency_for_no_clustering_result():
    """
    Test that the `utils.frequency.compute_clusters_frequency` works for no `clustering_result`.
    """

    # Check for no `clustering_result`.
    assert not compute_clusters_frequency(clustering_result={})


# ==============================================================================
# test_compute_clusters_frequency_for_correct_clustering_result
# ==============================================================================
def test_compute_clusters_frequency_for_correct_clustering_result():
    """
    Test that the `utils.frequency.compute_clusters_frequency` works for correct `clustering_result`.
    """

    # Check for no `clustering_result`.
    assert compute_clusters_frequency(
        clustering_result={
            "01": 0,
            "02": 0,
            "03": 0,
            "04": 0,
            "05": 0,
            "06": 1,
            "07": 1,
            "08": 2,
            "09": 2,
            "10": 2,
        }
    ) == {
        0: 0.5,
        1: 0.2,
        2: 0.3,
    }
