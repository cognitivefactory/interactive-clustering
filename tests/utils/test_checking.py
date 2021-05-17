# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/utils/test_checking.py
* Description:  Unittests for the `utils.checking` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
from cognitivefactory.interactive_clustering.utils.checking import (
    check_clustering_result,
    check_constraints_manager,
    check_vectors,
)


# ==============================================================================
# test_check_vectors_for_not_list_of_data_IDs
# ==============================================================================
def test_check_vectors_for_not_list_of_data_IDs():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for a not list `list_of_data_IDs`.
    """

    # Check for not list `list_of_data_IDs`.
    with pytest.raises(ValueError, match="`list_of_data_IDs`"):
        check_vectors(
            list_of_data_IDs="bonjour",
            vectors=None,
        )


# ==============================================================================
# test_check_vectors_for_not_dict_vectors
# ==============================================================================
def test_check_vectors_for_not_dict_vectors():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for a not dictionary `vectors`.
    """

    # Check for not dictionary `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
            ],
            vectors="ça ne marche pas",
        )


# ==============================================================================
# test_check_vectors_for_not_same_data_IDs_in_vectors
# ==============================================================================
def test_check_vectors_for_not_same_data_IDs_in_vectors():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for not same data IDs in `vectors`.
    """

    # Check for not same data IDs in `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
            ],
            vectors={
                "hello": None,
            },
        )


# ==============================================================================
# test_check_vectors_for_too_small_vectors
# ==============================================================================
def test_check_vectors_for_too_small_vectors():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for a too small `vectors` dictionary.
    """

    # Check for no two small `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
            ],
            vectors={
                "bonjour": None,
            },
        )


# ==============================================================================
# test_check_vectors_for_vectors_of_incorrect_type
# ==============================================================================
def test_check_vectors_for_vectors_of_incorrect_type():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for an incorrect `vectors` type.
    """

    # Check for `None` type in `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": None,
                "salut": None,
                "coucou": None,
            },
        )

    # Check for not allowed type in `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": "18",
                "salut": "14",
                "coucou": "32",
            },
        )


# ==============================================================================
# test_check_vectors_for_vectors_in_too_big_array
# ==============================================================================
def test_check_vectors_for_vectors_in_too_big_array():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for a too big dimensionned ndarray in `vectors`.
    """

    # Check for a too big dimensionned array in `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": np.array([[[[[1, 2, 3]]]]]),
                "salut": np.array([[[[[1, 2, 3]]]]]),
                "coucou": np.array([[[[[1, 2, 3]]]]]),
            },
        )


# ==============================================================================
# test_check_vectors_for_multiple_vectors_for_one_data_ID
# ==============================================================================
def test_check_vectors_for_multiple_vectors_for_one_data_ID():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` if one data ID has multiple vector in its `vectors` value.
    """

    # Check for a multiple row array in `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
                "salut": np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
                "coucou": np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
            },
        )

    # Check for a multiple row array in `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": csr_matrix(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
                "salut": csr_matrix(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
                "coucou": csr_matrix(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ),
            },
        )


# ==============================================================================
# test_check_vectors_for_different_size_of_vectors
# ==============================================================================
def test_check_vectors_for_different_size_of_vectors():
    """
    Test that the `utils.checking.check_vectors` method raise `ValueError` for vectors of different size.
    """

    # Check for `vectors` of different size.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": np.array([[1, 2, 3]]),
                "salut": np.array([[1, 2, 3, 4]]),
                "coucou": np.array([[1, 2, 3, 4, 5]]),
            },
        )

    # Check for `vectors` of different size.
    with pytest.raises(ValueError, match="`vectors`"):
        check_vectors(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            vectors={
                "bonjour": csr_matrix([1, 2, 3]),
                "salut": csr_matrix([1, 2, 3, 4]),
                "coucou": csr_matrix([1, 2, 3, 4, 5]),
            },
        )


# ==============================================================================
# test_check_vectors_for_correct_vectors
# ==============================================================================
def test_check_vectors_for_correct_vectors():
    """
    Test that the `utils.checking.check_vectors` method works for correct `vectors`.
    """

    # Check for `vectors`.
    new_vectors = check_vectors(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
        ],
        vectors={
            "bonjour": np.array([1, 2, 3]),
            "salut": np.array([[1, 2, 3]]),
            "coucou": csr_matrix([1, 2, 3]),
        },
    )

    assert new_vectors["bonjour"].shape == (1, 3)
    assert new_vectors["salut"].shape == (1, 3)
    assert new_vectors["coucou"].shape == (1, 3)


# ==============================================================================
# test_check_constraints_manager_for_not_list_of_data_IDs
# ==============================================================================
def test_check_constraints_manager_for_not_list_of_data_IDs():
    """
    Test that the `utils.checking.check_constraints_manager` method raise `ValueError` for a not list `list_of_data_IDs`.
    """

    # Check for not list `list_of_data_IDs`.
    with pytest.raises(ValueError, match="`list_of_data_IDs`"):
        check_constraints_manager(
            list_of_data_IDs="bonjour",
        )


# ==============================================================================
# test_check_constraints_manager_for_None_constraints_manager
# ==============================================================================
def test_check_constraints_manager_for_None_constraints_manager():
    """
    Test that the `utils.checking.check_constraints_manager` method works for no `constraints_manager`.
    """

    # Check for no `constraints_manager`.
    new_constraints_manager = check_constraints_manager(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
        ],
        constraints_manager=None,
    )

    assert isinstance(new_constraints_manager, AbstractConstraintsManager)
    assert new_constraints_manager.get_list_of_managed_data_IDs() == ["bonjour", "coucou", "salut"]


# ==============================================================================
# test_check_constraints_manager_for_not_AbstractConstraintsManager_constraints_manager
# ==============================================================================
def test_check_constraints_manager_for_not_AbstractConstraintsManager_constraints_manager():
    """
    Test that the `utils.checking.check_constraints_manager` method raise `ValueError` for not `AbstractConstraintsManager` `constraints_manager`.
    """

    # Check for not `AbstractConstraintsManager` `constraints_manager`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        check_constraints_manager(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            constraints_manager="constraints_manager",
        )


# ==============================================================================
# test_check_constraints_manager_for_not_same_data_IDs_in_constraints_manager
# ==============================================================================
def test_check_constraints_manager_for_not_same_data_IDs_in_constraints_manager():
    """
    Test that the `utils.checking.check_constraints_manager` method raise `ValueError` for not same data IDs in `constraints_manager`.
    """

    # Check for not same data IDs in `constraints_manager`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        check_constraints_manager(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ],
            constraints_manager=BinaryConstraintsManager(
                list_of_data_IDs=[
                    "hello",
                    "hi",
                ]
            ),
        )


# ==============================================================================
# test_check_constraints_manager_for_correct_constraints_manager
# ==============================================================================
def test_check_constraints_manager_for_correct_constraints_manager():
    """
    Test that the `utils.checking.check_constraints_manager` works for correct `constraints_manager`.
    """

    # Check for correct `constraints_manager`.
    new_constraints_manager = check_constraints_manager(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
        ],
        constraints_manager=BinaryConstraintsManager(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
            ]
        ),
    )

    assert isinstance(new_constraints_manager, AbstractConstraintsManager)
    assert new_constraints_manager.get_list_of_managed_data_IDs() == ["bonjour", "coucou", "salut"]


# ==============================================================================
# test_check_clustering_result_for_not_list_of_data_IDs
# ==============================================================================
def test_check_clustering_result_for_not_list_of_data_IDs():
    """
    Test that the `utils.checking.check_clustering_result` method raise `ValueError` for a not list `list_of_data_IDs`.
    """

    # Check for not list `list_of_data_IDs`.
    with pytest.raises(ValueError, match="`list_of_data_IDs`"):
        check_clustering_result(
            list_of_data_IDs="bonjour",
            clustering_result=None,
        )


# ==============================================================================
# test_check_clustering_result_for_not_dict_clustering_result
# ==============================================================================
def test_check_clustering_result_for_not_dict_clustering_result():
    """
    Test that the `utils.checking.check_clustering_result` method raise `ValueError` for a not dictionary `clustering_result`.
    """

    # Check for not dictionary `clustering_result`.
    with pytest.raises(ValueError, match="`clustering_result`"):
        check_clustering_result(
            list_of_data_IDs=[
                "bonjour",
            ],
            clustering_result="ça ne marche pas",
        )


# ==============================================================================
# test_check_clustering_result_for_not_same_data_IDs_in_clustering_result
# ==============================================================================
def test_check_clustering_result_for_not_same_data_IDs_in_clustering_result():
    """
    Test that the `utils.checking.check_clustering_result` method raise `ValueError` for not same data IDs in `clustering_result`.
    """

    # Check for not same data IDs in `clustering_result`.
    with pytest.raises(ValueError, match="`clustering_result`"):
        check_clustering_result(
            list_of_data_IDs=[
                "bonjour",
            ],
            clustering_result={
                "hello": None,
            },
        )


# ==============================================================================
# test_check_clustering_result_for_not_integer_clustering_ids
# ==============================================================================
def test_check_clustering_result_for_not_integer_clustering_ids():
    """
    Test that the `utils.checking.check_clustering_result` method raise `ValueError` for not interger cluster ids.
    """

    # Check for not interger cluster ids.
    with pytest.raises(ValueError, match="`clustering_result`"):
        check_clustering_result(
            list_of_data_IDs=["bonjour", "salut"],
            clustering_result={"bonjour": 1, "salut": "blue"},
        )


# ==============================================================================
# test_check_clustering_result_for_correct_clustering_result
# ==============================================================================
def test_check_clustering_result_for_correct_clustering_result():
    """
    Test that the `utils.checking.check_clustering_result` works for correct `clustering_result`.
    """

    # Check for correct `clustering_result`.
    new_clustering_result = check_clustering_result(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
        ],
        clustering_result={
            "bonjour": 1,
            "salut": 2,
            "coucou": 1,
        },
    )

    assert sorted(new_clustering_result.keys()) == ["bonjour", "coucou", "salut"]
