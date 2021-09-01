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

import pytest

from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
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
# test_RandomConstraintsSampling_sample_for_incorrect_constraints_manager
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_incorrect_constraints_manager():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling raises `ValueError` for incorrect `constraints_manager`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with incorrect `constraints_manager`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        sampler.sample(
            constraints_manager=None,
            nb_to_select=None,
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

    # Check sample with incorrect `nb_to_select`.
    with pytest.raises(ValueError, match="`nb_to_select`"):
        sampler.sample(
            constraints_manager=BinaryConstraintsManager(
                list_of_data_IDs=[
                    "bonjour",
                    "salut",
                    "coucou",
                    "au revoir",
                    "a bientôt",
                ]
            ),
            nb_to_select=None,
        )

    # Check sample with incorrect `nb_to_select`
    with pytest.raises(ValueError, match="`nb_to_select`"):
        sampler.sample(
            constraints_manager=BinaryConstraintsManager(
                list_of_data_IDs=[
                    "bonjour",
                    "salut",
                    "coucou",
                    "au revoir",
                    "a bientôt",
                ],
            ),
            nb_to_select=-99,
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
        constraints_manager=BinaryConstraintsManager(
            list_of_data_IDs=[
                "bonjour",
                "salut",
                "coucou",
                "au revoir",
                "a bientôt",
            ],
        ),
        nb_to_select=0,
    )


# ==============================================================================
# test_RandomConstraintsSampling_sample_for_empty_constraints_manager
# ==============================================================================
def test_RandomConstraintsSampling_sample_for_empty_constraints_manager():
    """
    Test that the `sampling.random.RandomConstraintsSampling` sampling works for empty `constraints_manager`.
    """

    # Initialize a `RandomConstraintsSampling` instance.
    sampler = RandomConstraintsSampling(random_seed=1)

    # Check sample with empty `constraints_manager`
    assert (
        sampler.sample(
            constraints_manager=BinaryConstraintsManager(
                list_of_data_IDs=[
                    "bonjour",
                    "salut",
                    "coucou",
                    "au revoir",
                    "a bientôt",
                ],
            ),
            nb_to_select=3,
        )
        == [("coucou", "salut"), ("bonjour", "coucou"), ("au revoir", "salut")]
    )


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
    assert (
        sampler.sample(
            constraints_manager=constraints_manager,
            nb_to_select=3,
        )
        == [("au revoir", "bonjour"), ("au revoir", "coucou"), ("bonjour", "coucou")]
    )


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
        constraints_manager=constraints_manager,
        nb_to_select=3,
    )
