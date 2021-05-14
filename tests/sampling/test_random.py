# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/sampling/test_random.py
* Description:  Unittests for the `sampling.random` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Needed library to apply tests.
import pytest

# Dependency needed to manage constraints.
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager

# Modules/Classes/Methods to test.
from cognitivefactory.interactive_clustering.sampling.random import RandomConstraintsSampling


# ==============================================================================
# test_RandomConstraintsSampling_for_correct_settings
# ==============================================================================
def test_RandomConstraintsSampling_for_correct_settings():
    """
    Test that the `sampling.random.RandomConstraintsSampling` works for correct settings.
    """

    # Check a correct initialization.
    sampler = RandomConstraintsSampling(random_seed=1)

    assert sampler
    assert sampler.random_seed == 1


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_incorrect_list_of_data_IDs
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_incorrect_list_of_data_IDs():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling raises `ValueError` for incorrect `list_of_data_IDs`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with incorrect `list_of_data_IDs`
    with pytest.raises(ValueError, match="`list_of_data_IDs`"):
        sampler.sample(
            list_of_data_IDs="unknown",
            nb_to_select=None,
            constraints_manager=None,
        )


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_incorrect_nb_to_select
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_incorrect_nb_to_select():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling raises `ValueError` for incorrect `nb_to_select`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with incorrect `nb_to_select`
    with pytest.raises(ValueError, match="`nb_to_select`"):
        sampler.sample(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
                "au revoir",
                "a bientôt",
            ],
            nb_to_select=None,
            constraints_manager=None,
        )

    # Check sample with incorrect `nb_to_select`
    with pytest.raises(ValueError, match="`nb_to_select`"):
        sampler.sample(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
                "au revoir",
                "a bientôt",
            ],
            nb_to_select=-99,
            constraints_manager=None,
        )


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_zero_nb_to_select
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_zero_nb_to_select():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling works for zero `nb_to_select`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with zero `nb_to_select`
    assert not sampler.sample(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
            "au revoir",
            "a bientôt",
        ],
        nb_to_select=0,
        constraints_manager=None,
    )


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_incorrect_constraints_manager
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_incorrect_constraints_manager():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling raises `ValueError` for incorrect `constraints_manager`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with incorrect `constraints_manager`
    with pytest.raises(ValueError, match="`constraints_manager`"):
        sampler.sample(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
                "au revoir",
                "a bientôt",
            ],
            nb_to_select=3,
            constraints_manager="unknown",
        )


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_no_constraints_manager
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_no_constraints_manager():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling works for no `constraints_manager`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with no `constraints_manager`
    assert sampler.sample(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
            "au revoir",
            "a bientôt",
        ],
        nb_to_select=3,
        constraints_manager=None,
    ) == [
        ("au revoir", "salut"),
        ("bonjour", "salut"),
        ("coucou", "salut"),
    ]


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_correct_constraints_manager
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_correct_constraints_manager():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling works for correct `constraints_manager`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Initialize a `BinaryConstraintsManager` instance
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
            "au revoir",
            "a bientôt",
        ]
    )
    constraints_manager.add_constraint(data_ID1="bonjour", data_ID2="salut", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="au revoir", data_ID2="a bientôt", constraint_type="MUST_LINK")

    # Check sample with correct `constraints_manager`
    assert sampler.sample(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
            "au revoir",
            "a bientôt",
        ],
        nb_to_select=3,
        constraints_manager=constraints_manager,
    ) == [
        ("au revoir", "bonjour"),
        ("bonjour", "coucou"),
        ("a bientôt", "coucou"),
    ]


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_full_annotated_constraints_manager
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_full_annotated_constraints_manager():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling works for full annotated `constraints_manager`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Initialize a `BinaryConstraintsManager` instance
    constraints_manager = BinaryConstraintsManager(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
            "au revoir",
            "a bientôt",
        ]
    )
    constraints_manager.add_constraint(data_ID1="bonjour", data_ID2="salut", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="bonjour", data_ID2="coucou", constraint_type="MUST_LINK")
    constraints_manager.add_constraint(data_ID1="bonjour", data_ID2="au revoir", constraint_type="CANNOT_LINK")
    constraints_manager.add_constraint(data_ID1="au revoir", data_ID2="a bientôt", constraint_type="MUST_LINK")

    # Check sample for full annotated `constraints_manager`
    assert not sampler.sample(
        list_of_data_IDs=[
            "bonjour",
            "salut",
            "coucou",
            "au revoir",
            "a bientôt",
        ],
        nb_to_select=3,
        constraints_manager=constraints_manager,
    )
