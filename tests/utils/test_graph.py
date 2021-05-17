# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/utils/test_graph.py
* Description:  Unittests for the `utils.graph` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import math

import pytest

from cognitivefactory.interactive_clustering.utils.graph import compute_lovasz_theta_number


# ==============================================================================
# test_compute_lovasz_theta_number_for_no_graph
# ==============================================================================
def test_compute_lovasz_theta_number_for_no_graph():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method for no graph raises `ValueError`.
    """

    ###
    ### No graph.
    ###

    # Define no graph.
    number_of_vertex = 0
    list_of_egdes = []

    # Compute Lovàsz number.
    with pytest.raises(ValueError, match="`number_of_vertex`"):
        compute_lovasz_theta_number(number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes)


# ==============================================================================
# test_compute_lovasz_theta_number_with_duplicates_in_list_of_edges
# ==============================================================================
def test_compute_lovasz_theta_number_with_duplicates_in_list_of_edges():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method raises `ValueError` in `list_of_edges` has duplicates.
    """

    # Define a graph with duplicates.
    number_of_vertex = 5
    list_of_egdes = [
        (1, 2),
        (2, 3),
        (2, 3),
    ]

    # Compute Lovàsz number.
    with pytest.raises(ValueError, match="`list_of_egdes`"):
        compute_lovasz_theta_number(number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes)


# ==============================================================================
# test_compute_lovasz_theta_number_with_not_ordered_edges
# ==============================================================================
def test_compute_lovasz_theta_number_with_not_ordered_edges():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method raises `ValueError` for `list_of_edges` with not ordered vertex.
    """

    # Define a graph.
    number_of_vertex = 5
    list_of_egdes = [
        (1, 2),
        (3, 2),
    ]

    # Compute Lovàsz number.
    with pytest.raises(ValueError, match="`list_of_egdes`"):
        compute_lovasz_theta_number(number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes)


# ==============================================================================
# test_compute_lovasz_theta_number_with_vertex_not_in_range
# ==============================================================================
def test_compute_lovasz_theta_number_with_vertex_not_in_range():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method raises `ValueError` for `list_of_edges` with not iun range vertex.
    """

    # Define a graph.
    number_of_vertex = 5
    list_of_egdes = [
        (1, 2),
        (1, 99999),
    ]

    # Compute Lovàsz number.
    with pytest.raises(ValueError, match="`list_of_egdes`"):
        compute_lovasz_theta_number(number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes)

    # Define a graph.
    number_of_vertex = 5
    list_of_egdes = [
        (1, 2),
        (-99999, 2),
    ]

    # Compute Lovàsz number.
    with pytest.raises(ValueError, match="`list_of_egdes`"):
        compute_lovasz_theta_number(number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes)


# ==============================================================================
# test_compute_lovasz_theta_number_for_singleton_graph
# ==============================================================================
def test_compute_lovasz_theta_number_for_singleton_graph():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method for a singleton graph works.
    """

    ###
    ### Singleton graph.
    ###

    # Define a singleton graph.
    number_of_vertex = 1
    list_of_egdes = []

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # theta == 1.0

    assert math.isclose(lovasz_results["theta"], 1, abs_tol=1e-5)
    assert lovasz_results["solver"] is None


# ==============================================================================
# test_compute_lovasz_theta_number_for_complete_graph
# ==============================================================================
def test_compute_lovasz_theta_number_for_complete_graph():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method for a complete graph works.
    Cf. https://en.wikipedia.org/wiki/Lov%C3%A1sz_number#Value_of_%CF%91_for_some_well-known_graphs
    """

    ###
    ### Complete graph of 5 vertex.
    ###

    # Define a complete graph.
    number_of_vertex = 5
    list_of_egdes = [(i, j) for i in range(number_of_vertex - 1) for j in range(i + 1, number_of_vertex)]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # theta == 1

    assert math.isclose(lovasz_results["theta"], 1, abs_tol=1e-5)
    assert lovasz_results["solver"]

    ###
    ### Complete graph of 10 vertex.
    ###

    # Define a complete graph.
    number_of_vertex = 10
    list_of_egdes = [(i, j) for i in range(number_of_vertex - 1) for j in range(i + 1, number_of_vertex)]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == 1`.

    assert math.isclose(lovasz_results["theta"], 1, abs_tol=1e-5)
    assert lovasz_results["solver"]


# ==============================================================================
# test_compute_lovasz_theta_number_for_empty_graph
# ==============================================================================
def test_compute_lovasz_theta_number_for_empty_graph():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method for an empty graph works.
    Cf. https://en.wikipedia.org/wiki/Lov%C3%A1sz_number#Value_of_%CF%91_for_some_well-known_graphs
    """

    ###
    ### Empty graph of 5 vertex.
    ###

    # Define an empty graph.
    number_of_vertex = 5
    list_of_egdes = []

    # Compute Lovàsz number
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == number_of_vertex == 5`.

    assert math.isclose(lovasz_results["theta"], 5, abs_tol=1e-5)
    assert lovasz_results["solver"]

    ###
    ### Empty graph of 10 vertex.
    ###

    # Define an empty graph.
    number_of_vertex = 10
    list_of_egdes = []

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == number_of_vertex == 10`.

    assert math.isclose(lovasz_results["theta"], 10, abs_tol=1e-5)
    assert lovasz_results["solver"]


# ==============================================================================
# test_compute_lovasz_theta_number_for_kneser_graph
# ==============================================================================
def test_compute_lovasz_theta_number_for_kneser_graph():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method for a kneser graph works.
    Cf. https://en.wikipedia.org/wiki/Lov%C3%A1sz_number#Value_of_%CF%91_for_some_well-known_graphs
    """

    ###
    ### Kneser graph KG_5,2 of 10 vertex.
    ### - A vertex can be representend by a couple of integers.
    ### - An edge link vertex that have disjoint integers in their couples.
    ###

    # Define a kneser graph.
    knerser_n = 5
    kneser_vertex = [[x1, x2] for x1 in range(knerser_n - 1) for x2 in range(x1 + 1, knerser_n)]
    number_of_vertex = len(kneser_vertex)
    list_of_egdes = [
        (i, j)
        for i, vi in enumerate(kneser_vertex)
        for j, vj in enumerate(kneser_vertex)
        if (i < j) and (len(set(vi + vj)) == len(vi + vj))
    ]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == math.comb(5-1,2-1) == 4`.

    assert math.isclose(lovasz_results["theta"], 4, abs_tol=1e-5)
    assert lovasz_results["solver"]

    ###
    ### Kneser graph KG_10,3 of 10 vertex.
    ### - A vertex can be representend by a triplet of integers.
    ### - An edge link vertex that have disjoint integers in their triplets.
    ###

    # Define a kneser graph.
    knerser_n = 10
    kneser_vertex = [
        [x1, x2, x3]
        for x1 in range(knerser_n - 2)
        for x2 in range(x1 + 1, knerser_n - 1)
        for x3 in range(x2 + 1, knerser_n)
    ]
    number_of_vertex = len(kneser_vertex)
    list_of_egdes = [
        (i, j)
        for i, vi in enumerate(kneser_vertex)
        for j, vj in enumerate(kneser_vertex)
        if (i < j) and (len(set(vi + vj)) == len(vi + vj))
    ]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == math.comb(10-1,3-1) == 36`.

    assert math.isclose(lovasz_results["theta"], 36, abs_tol=1e-5)
    assert lovasz_results["solver"]


# ==============================================================================
# test_compute_lovasz_theta_number_for_cycle_graph
# ==============================================================================
def test_compute_lovasz_theta_number_for_cycle_graph():
    """
    Test that the `utils.graph.compute_lovasz_theta_number` method for a cycle graph works.
    Cf. https://en.wikipedia.org/wiki/Lov%C3%A1sz_number#Value_of_%CF%91_for_some_well-known_graphs
    """

    ###
    ### Cycle graph of 5 vertex (pentagon case).
    ###

    # Define a cycle graph.
    number_of_vertex = 5
    list_of_egdes = [(i, i + 1) for i in range(number_of_vertex - 1)] + [(0, number_of_vertex - 1)]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == math.sqrt(5) == (5 * math.cos(math.pi/5)) / (1+math.cos(math.pi/5)) ~= 2.2360`.

    theoric_value = (5 * math.cos(math.pi / 5)) / (1 + math.cos(math.pi / 5))
    assert math.isclose(lovasz_results["theta"], theoric_value, abs_tol=1e-5)
    assert lovasz_results["solver"]

    ###
    ### Cycle graph of 5 vertex (odd case).
    ###

    # Define a cycle graph.
    number_of_vertex = 7
    list_of_egdes = [(i, i + 1) for i in range(number_of_vertex - 1)] + [(0, number_of_vertex - 1)]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == (7 * math.cos(math.pi/7)) / (1+math.cos(math.pi/7)) ~= 3.3176`.

    theoric_value = (7 * math.cos(math.pi / 7)) / (1 + math.cos(math.pi / 7))
    assert math.isclose(lovasz_results["theta"], theoric_value, abs_tol=1e-5)
    assert lovasz_results["solver"]

    ###
    ### Cycle graph of 10 vertex (even case).
    ###

    # Define a cycle graph.
    number_of_vertex = 10
    list_of_egdes = [(i, i + 1) for i in range(number_of_vertex - 1)] + [(0, number_of_vertex - 1)]

    # Compute Lovàsz number.
    lovasz_results = compute_lovasz_theta_number(
        number_of_vertex=number_of_vertex, list_of_egdes=list_of_egdes
    )  # `theta == 10/2 == 5`.

    assert math.isclose(lovasz_results["theta"], 5, abs_tol=1e-5)
    assert lovasz_results["solver"]
