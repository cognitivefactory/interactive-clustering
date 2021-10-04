# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/sampling/test_factory_farthest_in_same_cluster.py
* Description:  Unittests for the `sampling.cluster_based` module, `"farhest_in_same_cluster"` sampler.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
from cognitivefactory.interactive_clustering.sampling.clusters_based import ClustersBasedConstraintsSampling


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_for_correct_settings
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_for_correct_settings():
    """
    Test that the `farhest_in_same_cluster sampler` works for correct settings.
    """

    # Check a correct initialization.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

    assert sampler
    assert sampler.random_seed == 1


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_constraints_manager
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_constraints_manager():
    """
    Test that the `farhest_in_same_cluster sampler` sampling raises `ValueError` for incorrect `constraints_manager`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

    # Check sample with incorrect `constraints_manager`.
    with pytest.raises(ValueError, match="`constraints_manager`"):
        sampler.sample(
            constraints_manager=None,
            nb_to_select=None,
        )


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_nb_to_select
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_nb_to_select():
    """
    Test that the `farhest_in_same_cluster sampler` sampling raises `ValueError` for incorrect `nb_to_select`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

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
# test_factory_farhest_in_same_cluster_sampler_sample_for_zero_nb_to_select
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_zero_nb_to_select():
    """
    Test that the `farhest_in_same_cluster sampler` sampling works for zero `nb_to_select`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

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
# test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_clustering_result
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_clustering_result():
    """
    Test that the `farhest_in_same_cluster sampler` sampling raises `ValueError` for incorrect `clustering_result`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

    # Check sample with incorrect `clustering_result`.
    with pytest.raises(ValueError, match="`clustering_result`"):
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
            clustering_result="unknown",
        )

    # Check sample with incorrect `clustering_result`.
    with pytest.raises(KeyError, match="'a bientôt'|'au revoir'|'bonjour'|'coucou'|'salut'"):
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
            clustering_result={
                "first": 1,
                "second": 2,
            },
            vectors={
                "bonjour": csr_matrix([1.0, 0.0]),
                "salut": csr_matrix([0.99, 0.0]),
                "coucou": csr_matrix([0.8, 0.0]),
                "au revoir": csr_matrix([0.0, 1.0]),
                "a bientôt": csr_matrix([0.0, 0.9]),
            },
        )


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_vectors
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_incorrect_vectors():
    """
    Test that the `farhest_in_same_cluster sampler` sampling raises `ValueError` for incorrect `vectors`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

    # Check sample with incorrect `vectors`.
    with pytest.raises(ValueError, match="`vectors`"):
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
            clustering_result={
                "bonjour": 0,
                "salut": 0,
                "coucou": 0,
                "au revoir": 1,
                "a bientôt": 1,
            },
            vectors="unknown",
        )

    # Check sample with incorrect `vectors`.
    with pytest.raises(KeyError, match="'a bientôt'|'au revoir'|'bonjour'|'coucou'|'salut'"):
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
            clustering_result={
                "bonjour": 0,
                "salut": 0,
                "coucou": 0,
                "au revoir": 1,
                "a bientôt": 1,
            },
            vectors={
                "first": 1,
                "second": 2,
            },
        )


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_sample_for_empty_constraints_manager
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_empty_constraints_manager():
    """
    Test that the `farhest_in_same_cluster sampler` sampling works for empty `constraints_manager`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

    # Check sample with empty `constraints_manager`
    assert sampler.sample(
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
        clustering_result={
            "bonjour": 0,
            "salut": 0,
            "coucou": 0,
            "au revoir": 1,
            "a bientôt": 1,
        },
        vectors={
            "bonjour": csr_matrix([1.0, 0.0]),
            "salut": csr_matrix([0.99, 0.0]),
            "coucou": csr_matrix([0.8, 0.0]),
            "au revoir": csr_matrix([0.0, 0.9]),
            "a bientôt": csr_matrix([0.0, 0.8]),
        },
    ) == [
        ("bonjour", "coucou"),
        ("coucou", "salut"),
        ("a bientôt", "au revoir"),
    ]


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_sample_for_correct_constraints_manager
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_correct_constraints_manager():
    """
    Test that the `farhest_in_same_cluster sampler` sampling works for correct `constraints_manager`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

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
        constraints_manager=constraints_manager,
        nb_to_select=3,
        clustering_result={
            "bonjour": 0,
            "salut": 0,
            "coucou": 0,
            "au revoir": 1,
            "a bientôt": 1,
        },
        vectors={
            "bonjour": csr_matrix([1.0, 0.0]),
            "salut": csr_matrix([0.99, 0.0]),
            "coucou": csr_matrix([0.8, 0.0]),
            "au revoir": csr_matrix([0.0, 0.9]),
            "a bientôt": csr_matrix([0.0, 0.8]),
        },
    ) == [
        ("bonjour", "coucou"),
        ("coucou", "salut"),
    ]


# ==============================================================================
# test_factory_farhest_in_same_cluster_sampler_sample_for_full_annotated_constraints_manager
# ==============================================================================
def test_factory_farhest_in_same_cluster_sampler_sample_for_full_annotated_constraints_manager():
    """
    Test that the `farhest_in_same_cluster sampler` sampling works for full annotated `constraints_manager`.
    """

    # Initialize a `farhest_in_same_cluster sampler` instance.
    sampler = ClustersBasedConstraintsSampling(
        clusters_restriction="same_cluster",
        distance_restriction="farthest_neighbors",
        random_seed=1,
    )

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
        clustering_result={
            "bonjour": 0,
            "salut": 0,
            "coucou": 0,
            "au revoir": 1,
            "a bientôt": 1,
        },
        vectors={
            "bonjour": csr_matrix([1.0, 0.0]),
            "salut": csr_matrix([0.99, 0.0]),
            "coucou": csr_matrix([0.8, 0.0]),
            "au revoir": csr_matrix([0.0, 0.9]),
            "a bientôt": csr_matrix([0.0, 0.8]),
        },
    )
