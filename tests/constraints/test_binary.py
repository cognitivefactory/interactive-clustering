# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/constraints/test_binary.py
* Description:  Unittests for the `constraints.binary` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# test_BinaryConstraintsManager_init
# ==============================================================================
def test_BinaryConstraintsManager_init():
    """
    Test that the `__init__` method of the `constraints.binary.BinaryConstraintsManager` class works.
    """

    # Initialize an empty binaray constraints manager.
    constraints_manager_1 = BinaryConstraintsManager(
        list_of_data_IDs=[],
    )
    assert not constraints_manager_1.get_list_of_managed_data_IDs()

    # Initialize a classic binaray constraints manager.
    constraints_manager_2 = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )
    assert constraints_manager_2.get_list_of_managed_data_IDs() == ["first", "second", "third"]

    # Initialize a binaray constraints manager with duplicates in `list_of_data_IDs`.
    with pytest.raises(ValueError, match="`list_of_data_IDs`"):
        BinaryConstraintsManager(
            list_of_data_IDs=["first", "second", "same", "same"],
        )


# ==============================================================================
# test_BinaryConstraintsManager_add_data_ID_with_incorrect_data_ID
# ==============================================================================
def test_BinaryConstraintsManager_add_data_ID_with_incorrect_data_ID():
    """
    Test that the `add_data_ID` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for an incorrect data ID.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to add `"second"` data ID.
    with pytest.raises(ValueError, match="`data_ID`"):
        constraints_manager.add_data_ID(
            data_ID="second",
        )

    # Run assertions.
    assert constraints_manager.get_list_of_managed_data_IDs() == ["first", "second", "third"]


# ==============================================================================
# test_BinaryConstraintsManager_add_data_ID_with_correct_parameters
# ==============================================================================
def test_BinaryConstraintsManager_add_data_ID_with_correct_parameters():
    """
    Test that the `add_data_ID` method of the `constraints.binary.BinaryConstraintsManager` class returns `True` for a correct parameters.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to add `"fourth"` data ID.
    assert (
        constraints_manager.add_data_ID(
            data_ID="fourth",
        )
        is True
    )

    # Run assertions.
    assert constraints_manager.get_list_of_managed_data_IDs() == ["first", "second", "third", "fourth"]
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="second",
            data_ID2="fourth",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="fourth",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="fourth",
            data_ID2="fourth",
        )
        == "MUST_LINK"
    )


# ==============================================================================
# test_BinaryConstraintsManager_delete_data_ID_with_incorrect_data_ID
# ==============================================================================
def test_BinaryConstraintsManager_delete_data_ID_with_incorrect_data_ID():
    """
    Test that the `delete_data_ID` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for an incorrect data ID.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to delete `"unknown"` data ID.
    with pytest.raises(ValueError, match="`data_ID`"):
        constraints_manager.delete_data_ID(
            data_ID="unknown",
        )

    # Run assertions.
    assert constraints_manager.get_list_of_managed_data_IDs() == ["first", "second", "third"]


# ==============================================================================
# test_BinaryConstraintsManager_delete_data_ID_with_correct_parameters
# ==============================================================================
def test_BinaryConstraintsManager_delete_data_ID_with_correct_parameters():
    """
    Test that the `delete_data_ID` method of the `constraints.binary.BinaryConstraintsManager` class returns `True` for a correct parameters.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to delete `"second"` data ID.
    assert (
        constraints_manager.delete_data_ID(
            data_ID="second",
        )
        is True
    )

    # Run assertions.
    assert constraints_manager.get_list_of_managed_data_IDs() == ["first", "third"]


# ==============================================================================
# test_BinaryConstraintsManager_get_list_of_managed_data_IDs
# ==============================================================================

# Not implemented because `get_list_of_managed_data_IDs` is already tested in other unittests.


# ==============================================================================
# test_BinaryConstraintsManager_add_constraint_with_incorrect_data_ID
# ==============================================================================
def test_BinaryConstraintsManager_add_constraint_with_incorrect_data_ID():
    """
    Test that the `add_constraint` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for incorrect data IDs.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to add the constraint with `"unknown"` data ID in `"data_ID1"`.
    with pytest.raises(ValueError, match="`data_ID1`"):
        constraints_manager.add_constraint(
            data_ID1="unknown",
            data_ID2="second",
            constraint_type="MUST_LINK",
        )

    # Try to add the constraint with `"unknown"` data ID in `"data_ID2"`.
    with pytest.raises(ValueError, match="`data_ID2`"):
        constraints_manager.add_constraint(
            data_ID1="first",
            data_ID2="unknown",
            constraint_type="CANNOT_LINK",
        )


# ==============================================================================
# test_BinaryConstraintsManager_add_constraint_with_incorrect_constraint_type
# ==============================================================================
def test_BinaryConstraintsManager_add_constraint_with_incorrect_constraint_type():
    """
    Test that the `add_constraint` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for an incorrect constraint type.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to add the constraint with `"UNKNOWN_LINK"` constraint type in `"constraint_type"`.
    with pytest.raises(ValueError, match="`constraint_type`"):
        constraints_manager.add_constraint(
            data_ID1="first",
            data_ID2="second",
            constraint_type="UNKNOWN_LINK",
        )


# ==============================================================================
# test_BinaryConstraintsManager_add_constraint_with_already_linked_data_IDs
# ==============================================================================
def test_BinaryConstraintsManager_add_constraint_with_already_linked_data_IDs():
    """
    Test that the `add_constraint` method of the `constraints.binary.BinaryConstraintsManager` class works for already linked data IDs.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="third",
        constraint_type="CANNOT_LINK",
    )

    # Try to add `"MUST_LINK"` data ID between `"first"` and `"second"` data IDs.
    assert (
        constraints_manager.add_constraint(
            data_ID1="first",
            data_ID2="second",
            constraint_type="MUST_LINK",
        )
        is True
    )

    # Try to add `"CANNOT_LINK"` data ID between `"first"` and `"second"` data IDs.
    with pytest.raises(ValueError, match="`constraint_type`"):
        constraints_manager.add_constraint(
            data_ID1="first",
            data_ID2="second",
            constraint_type="CANNOT_LINK",
        )


# ==============================================================================
# test_BinaryConstraintsManager_add_constraint_with_not_already_linked_data_IDs
# ==============================================================================
def test_BinaryConstraintsManager_add_constraint_with_not_already_linked_data_IDs():
    """
    Test that the `add_constraint` method of the `constraints.binary.BinaryConstraintsManager` class works for not already linked data IDs.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to add `"MUST_LINK"` data ID between `"first"` and `"second"` data IDs.
    assert (
        constraints_manager.add_constraint(
            data_ID1="first",
            data_ID2="second",
            constraint_type="MUST_LINK",
        )
        is True
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        == ("MUST_LINK", 1.0)
    )

    # Try to add `"CANNOT_LINK"` data ID between `"first"` and `"third"` data IDs.
    assert (
        constraints_manager.add_constraint(
            data_ID1="first",
            data_ID2="third",
            constraint_type="CANNOT_LINK",
        )
        is True
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == ("CANNOT_LINK", 1.0)
    )

    # Try to add `"MUST_LINK"` data ID between `"second"` and `"third"` data IDs.
    with pytest.raises(ValueError, match="`constraint_type`"):
        constraints_manager.add_constraint(
            data_ID1="second",
            data_ID2="third",
            constraint_type="MUST_LINK",
        )

    # Try to add `"CANNOT_LINK"` data ID between `"second"` and `"third"` data IDs.
    assert (
        constraints_manager.add_constraint(
            data_ID1="second",
            data_ID2="third",
            constraint_type="CANNOT_LINK",
        )
        is True
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="second",
            data_ID2="third",
        )
        == ("CANNOT_LINK", 1.0)
    )


# ==============================================================================
# test_BinaryConstraintsManager_delete_constraint_with_incorrect_data_IDs
# ==============================================================================
def test_BinaryConstraintsManager_delete_constraint_with_incorrect_data_IDs():
    """
    Test that the `delete_constraint` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for incorrect data IDs.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try to delete constraint with `"unknown"` data ID in `"data_ID1"`.
    with pytest.raises(ValueError, match="`data_ID1`"):
        constraints_manager.delete_constraint(
            data_ID1="unknown",
            data_ID2="second",
        )

    # Try to delete constraint with `"unknown"` data ID in `"data_ID2"`.
    with pytest.raises(ValueError, match="`data_ID2`"):
        constraints_manager.delete_constraint(
            data_ID1="first",
            data_ID2="unknown",
        )


# ==============================================================================
# test_BinaryConstraintsManager_delete_constraint_with_correct_parameters
# ==============================================================================
def test_BinaryConstraintsManager_delete_constraint_with_correct_parameters():
    """
    Test that the `delete_constraint` method of the `constraints.binary.BinaryConstraintsManager` returns `True` for correct parameters.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="third",
        constraint_type="CANNOT_LINK",
    )

    # Try to delete constraint betwenn `"second"` and `"third"` data IDs.
    assert (
        constraints_manager.delete_constraint(
            data_ID1="second",
            data_ID2="third",
        )
        is True
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="second",
            data_ID2="third",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="second",
            data_ID2="third",
        )
        == "CANNOT_LINK"
    )  # transitivity activated !

    # Try to delete constraint betwenn `"first"` and `"third"` data IDs.
    assert (
        constraints_manager.delete_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        is True
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        is None
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="second",
            data_ID2="third",
        )
        is None
    )  # No more transitivity


# ==============================================================================
# test_BinaryConstraintsManager_get_added_constraint_with_incorrect_data_ID
# ==============================================================================
def test_BinaryConstraintsManager_get_added_constraint_with_incorrect_data_ID():
    """
    Test that the `get_added_constraint` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for an incorrect data ID.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="third",
        constraint_type="CANNOT_LINK",
    )

    # Try to get the constraint with `"unknown"` data ID in `"data_ID1"`.
    with pytest.raises(ValueError, match="`data_ID1`"):
        constraints_manager.get_added_constraint(
            data_ID1="unknown",
            data_ID2="second",
        )

    # Try to get the constraint with `"unknown"` data ID in `"data_ID2"`.
    with pytest.raises(ValueError, match="`data_ID2`"):
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="unknown",
        )


# ==============================================================================
# test_BinaryConstraintsManager_get_added_constraint_with_correct_parameters
# ==============================================================================
def test_BinaryConstraintsManager_get_added_constraint_with_correct_parameters():
    """
    Test that the `get_added_constraint` method of the `constraints.binary.BinaryConstraintsManager` class returns `True` for correct parameters.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="third",
        constraint_type="CANNOT_LINK",
    )

    # Try to get the constraint betwwen `"first"` and `"second"`.
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        == ("MUST_LINK", 1.0)
    )

    # Try to get the constraint betwwen `"first"` and `"third"`.
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == ("CANNOT_LINK", 1.0)
    )


# ==============================================================================
# test_BinaryConstraintsManager_get_inferred_constraint_with_incorrect_data_ID
# ==============================================================================
def test_BinaryConstraintsManager_get_inferred_constraint_with_incorrect_data_ID():
    """
    Test that the `get_inferred_constraint` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for an incorrect data ID.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try `get_inferred_constraint` method with `"unknown"` data ID in `data_ID1`.
    with pytest.raises(ValueError, match="`data_ID1`"):
        constraints_manager.get_inferred_constraint(
            data_ID1="unknown",
            data_ID2="second",
        )

    # Try `get_inferred_constraint` method with `"unknown"` data ID in `data_ID2`.
    with pytest.raises(ValueError, match="`data_ID2`"):
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="unknown",
        )


# ==============================================================================
# test_BinaryConstraintsManager_get_inferred_constraint_with_correct_parameter
# ==============================================================================
def test_BinaryConstraintsManager_get_inferred_constraint_with_correct_parameter():
    """
    Test that the `get_inferred_constraint` method of the `constraints.binary.BinaryConstraintsManager` class works for a correct data ID.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )

    # Try to check the link bewteen `"first"` and `"second"`data IDs.
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        == "MUST_LINK"
    )

    # Try to check the link bewteen `"first"` and `"third"`data IDs.
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        is None
    )


# ==============================================================================
# test_BinaryConstraintsManager_get_connected_components
# ==============================================================================
def test_BinaryConstraintsManager_get_connected_components():
    """
    Test that the `get_connected_components` method of the `constraints.binary.BinaryConstraintsManager` class works.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    # Add `"MUST_LINK"` constraint between `"second"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="MUST_LINK",
    )
    # Add `"CANNOT_LINK"` constraint between `"second"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )

    # Try to get the list of connected components.
    connected_component = constraints_manager.get_connected_components()
    assert connected_component == [["first", "second", "third"], ["fourth"]]


# ==============================================================================
# test_BinaryConstraintsManager_check_completude_of_constraints
# ==============================================================================
def test_BinaryConstraintsManager_check_completude_of_constraints():
    """
    Test that the `check_completude_of_constraints` method of the `constraints.binary.BinaryConstraintsManager` class works.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth"],
    )
    assert constraints_manager.check_completude_of_constraints() is False

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    assert constraints_manager.check_completude_of_constraints() is False

    # Add `"MUST_LINK"` constraint between `"second"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="MUST_LINK",
    )
    assert constraints_manager.check_completude_of_constraints() is False

    # Add `"CANNOT_LINK"` constraint between `"second"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )
    assert constraints_manager.check_completude_of_constraints() is True


# ==============================================================================
# test_BinaryConstraintsManager_get_min_and_max_number_of_clusters
# ==============================================================================
def test_BinaryConstraintsManager_get_min_and_max_number_of_clusters():
    """
    Test that the `get_min_and_max_number_of_clusters` method of the `constraints.binary.BinaryConstraintsManager` class works.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth", "fifth", "sixth"],
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 2
    assert max == 6

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 2
    assert max == 5

    # Add `"CANNOT_LINK"` constraint between `"second"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="CANNOT_LINK",
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 2
    assert max == 5

    # Add `"CANNOT_LINK"` constraint between `"second"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 2
    assert max == 5

    # Add `"CANNOT_LINK"` constraint between `"third"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="third",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 3  # 2.99999...
    assert max == 5

    # Add `"MUST_LINK"` constraint between `"fourth"` and `"fifth"`.
    constraints_manager.add_constraint(
        data_ID1="fourth",
        data_ID2="fifth",
        constraint_type="MUST_LINK",
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 3  # 2.99999...
    assert max == 4

    # Add `"MUST_LINK"` constraint between `"fourth"` and `"sixth"`.
    constraints_manager.add_constraint(
        data_ID1="fourth",
        data_ID2="sixth",
        constraint_type="MUST_LINK",
    )
    min, max = constraints_manager.get_min_and_max_number_of_clusters()
    assert min == 3  # 2.99999...
    assert max == 3
    assert constraints_manager.check_completude_of_constraints() is True


# ==============================================================================
# test_BinaryConstraintsManager_check_symetry_after_constraint_addition
# ==============================================================================
def test_BinaryConstraintsManager_check_symetry_after_constraint_addition():
    """
    Test that the `add_constraint` method of the `constraints.binary.BinaryConstraintsManager` class is symetric.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    # Add `"CANNOT_LINK"` constraint between `"first"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="third",
        constraint_type="CANNOT_LINK",
    )

    # Run assertion on : ML(1,2) => ML(2,1).
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        == ("MUST_LINK", 1.0)
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="second",
            data_ID2="first",
        )
        == ("MUST_LINK", 1.0)
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        == "MUST_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="second",
            data_ID2="first",
        )
        == "MUST_LINK"
    )

    # Run assertion on : CL(1,3) => CL(3,1).
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == ("CANNOT_LINK", 1.0)
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        == ("CANNOT_LINK", 1.0)
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == "CANNOT_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        == "CANNOT_LINK"
    )


# ==============================================================================
# test_BinaryConstraintsManager_check_transitivity_after_constraint_addition
# ==============================================================================
def test_BinaryConstraintsManager_check_transitivity_after_constraint_addition():
    """
    Test that the `add_constraint` method of the `constraints.binary.BinaryConstraintsManager` class is transitive.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    # Add `"MUST_LINK"` constraint between `"second"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="MUST_LINK",
    )
    # Add `"CANNOT_LINK"` constraint between `"second"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )

    # Run assertion on : ML(1,2)+ML(2,3) => ML(1,3) AND ML(3,1).
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        is None
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == "MUST_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        == "MUST_LINK"
    )

    # Run assertion on : ML(1,2)+CL(2,4) => CL(1,4) AND CL(4,1).
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        is None
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="fourth",
            data_ID2="first",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        == "CANNOT_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="fourth",
            data_ID2="first",
        )
        == "CANNOT_LINK"
    )


# ==============================================================================
# test_BinaryConstraintsManager_check_symetry_after_constraint_deletion
# ==============================================================================
def test_BinaryConstraintsManager_check_symetry_after_constraint_deletion():
    """
    Test that the `delete_constraint` method of the `constraints.binary.BinaryConstraintsManager` class is symetric.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    # Delete constraint between `"second"` and `"first"`.
    constraints_manager.delete_constraint(
        data_ID1="second",
        data_ID2="first",
    )

    # Run assertion (no more symetry).
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        is None
    )
    assert (
        constraints_manager.get_added_constraint(
            data_ID1="second",
            data_ID2="first",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="second",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="second",
            data_ID2="first",
        )
        is None
    )


# ==============================================================================
# test_BinaryConstraintsManager_check_transitivity_after_constraint_deletion
# ==============================================================================
def test_BinaryConstraintsManager_check_transitivity_after_constraint_deletion():
    """
    Test that the `delete_constraint` method of the `constraints.binary.BinaryConstraintsManager` class is transitive.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )

    # Add `"MUST_LINK"` constraint between `"second"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="MUST_LINK",
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == "MUST_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        == "MUST_LINK"
    )

    # Add `"CANNOT_LINK"` constraint between `"second"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        == "CANNOT_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="fourth",
            data_ID2="first",
        )
        == "CANNOT_LINK"
    )

    # Delete constraint between `"second"` and `"first"`.
    constraints_manager.delete_constraint(
        data_ID1="second",
        data_ID2="first",
    )

    # Run assertion (no more transitivity).
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="fourth",
            data_ID2="first",
        )
        is None
    )


# ==============================================================================
# test_BinaryConstraintsManager_check_transitivity_after_data_ID_deletion
# ==============================================================================
def test_BinaryConstraintsManager_check_transitivity_after_data_ID_deletion():
    """
    Test that the `delete_data_ID` method of the `constraints.binary.BinaryConstraintsManager` class is transitive.
    """

    # Initialize a classic binaray constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth"],
    )

    # Add `"MUST_LINK"` constraint between `"first"` and `"second"`.
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    # Add `"MUST_LINK"` constraint between `"second"` and `"third"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="MUST_LINK",
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        == "MUST_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        == "MUST_LINK"
    )

    # Add `"CANNOT_LINK"` constraint between `"second"` and `"fourth"`.
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        == "CANNOT_LINK"
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="fourth",
            data_ID2="first",
        )
        == "CANNOT_LINK"
    )

    # Delete data ID `"second"`.
    constraints_manager.delete_data_ID(
        data_ID="second",
    )

    # Run assertion (no more transitivity).
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="third",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="third",
            data_ID2="first",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="first",
            data_ID2="fourth",
        )
        is None
    )
    assert (
        constraints_manager.get_inferred_constraint(
            data_ID1="fourth",
            data_ID2="first",
        )
        is None
    )


# ==============================================================================
# test_BinaryConstraintsManager_get_list_of_involved_data_IDs_in_a_constraint_conflict_with_incorrect_data_ID
# ==============================================================================
def test_BinaryConstraintsManager_get_list_of_involved_data_IDs_in_a_constraint_conflict_with_incorrect_data_ID():
    """
    Test that the `get_list_of_involved_data_IDs_in_a_constraint_conflict` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for incorrect data IDs.
    """

    # Initialize a classic binary constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try with `"unknown"` data ID in `"data_ID1"`.
    with pytest.raises(ValueError, match="`data_ID1`"):
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="unknown",
            data_ID2="second",
            constraint_type="MUST_LINK",
        )

    # Try with `"unknown"` data ID in `"data_ID2"`.
    with pytest.raises(ValueError, match="`data_ID2`"):
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="first",
            data_ID2="unknown",
            constraint_type="CANNOT_LINK",
        )


# ==============================================================================
# test_BinaryConstraintsManager_get_list_of_involved_data_IDs_in_a_constraint_conflict_with_incorrect_constraint_type
# ==============================================================================
def test_BinaryConstraintsManager_get_list_of_involved_data_IDs_in_a_constraint_conflict_with_incorrect_constraint_type():
    """
    Test that the `get_list_of_involved_data_IDs_in_a_constraint_conflict` method of the `constraints.binary.BinaryConstraintsManager` class raises `ValueError` for an incorrect constraint type.
    """

    # Initialize a classic binary constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third"],
    )

    # Try with `"UNKNOWN_LINK"` constraint type in `"constraint_type"`.
    with pytest.raises(ValueError, match="`constraint_type`"):
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="first",
            data_ID2="second",
            constraint_type="UNKNOWN_LINK",
        )


# ==============================================================================
# test_BinaryConstraintsManager_get_list_of_involved_data_IDs_in_a_constraint_conflict
# ==============================================================================
def test_BinaryConstraintsManager_get_list_of_involved_data_IDs_in_a_constraint_conflict():
    """
    Test that the `get_list_of_involved_data_IDs_in_a_constraint_conflict` method of the `constraints.binary.BinaryConstraintsManager` class works.
    """

    # Initialize a classic binary constraints manager.
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=["first", "second", "third", "fourth", "fifth"],
    )
    constraints_manager.add_constraint(
        data_ID1="first",
        data_ID2="second",
        constraint_type="MUST_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="second",
        data_ID2="third",
        constraint_type="MUST_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="third",
        data_ID2="fourth",
        constraint_type="CANNOT_LINK",
    )
    constraints_manager.add_constraint(
        data_ID1="fourth",
        data_ID2="fifth",
        constraint_type="CANNOT_LINK",
    )

    # Conflict of adding `"MUST_LINK"` between "first" and "fourth".
    assert (
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="first",
            data_ID2="fourth",
            constraint_type="MUST_LINK",
        )
        == ["first", "second", "third", "fourth"]
    )

    # Conflict of adding `"CANNOT_LINK"` between "first" and "fourth".
    assert (
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="first",
            data_ID2="third",
            constraint_type="CANNOT_LINK",
        )
        == ["first", "second", "third"]
    )

    # No conflict.
    assert (
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="first",
            data_ID2="third",
            constraint_type="MUST_LINK",
        )
        is None
    )

    # No conflict.
    assert (
        constraints_manager.get_list_of_involved_data_IDs_in_a_constraint_conflict(
            data_ID1="fourth",
            data_ID2="fifth",
            constraint_type="CANNOT_LINK",
        )
        is None
    )
