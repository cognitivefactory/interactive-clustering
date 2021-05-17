# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.utils.graph
* Description:  Utilities methods for graph analysis.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORTS :
# =============================================================================

from typing import Any, Dict, List, Tuple  # To type Python code (mypy).

import cvxopt  # To solve semidefine programming.


# ==============================================================================
# COMPUTE LOVACZ THETA NUMBER
# ==============================================================================
def compute_lovasz_theta_number(
    number_of_vertex: int,
    list_of_egdes: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """
    Computes the Lovasz theta number for a graph.

    Args:
        number_of_vertex (int): The number of vertex in the graph.
        list_of_egdes (List[Tuple[int,int]]): The list of edges of the graph, with no duplicates, where each edge are modeled by an ordered couple of vertex.

    Raises:
        ValueError: if `number_of_vertex` is badly set (has to be greater than 0) or if `list_of_egdes` is badly set (has to be with no duplicates, with ordered tuple, with vertex in range of `number_of_vertex`).

    Returns:
        Dict[str, Any]: A dictionary of results with `"theta"` (the Lovasz theta number) and `"solver"` (the solver used to compute it).
    """

    # Get the number of edges
    number_of_edges: int = len(list_of_egdes)

    # Case of duplicates in `list_of_edges`.
    if number_of_edges != len(set(list_of_egdes)):
        raise ValueError("There is duplicates in `list_of_egdes`.")

    # Case of parameter `number_of_vertex` lower of equal to 0.
    if number_of_vertex <= 0:
        raise ValueError("The parameters `number_of_vertex` can't be lower or equal to `0`.")

    # Case of single vertex graph.
    if number_of_vertex == 1:
        return {
            "theta": 1.0,
            "solver": None,
        }

    # Initialization and definition of parameter `c` of the solver.
    c = cvxopt.matrix(x=[0.0 for _ in range(number_of_edges)] + [1.0])

    # Initialization and definition of parameter `G1` of the solver.
    G1 = cvxopt.spmatrix(
        V=0,  # default value of nonzero entries
        I=[],  # list of row indices
        J=[],  # list of column indices
        size=(number_of_vertex * number_of_vertex, number_of_edges + 1),  # number of rows  # number of columns
    )

    # Completion of parameter `G1` of the solver.
    for e, (v1, v2) in enumerate(list_of_egdes):

        # Check that vertex in egde are ordered.
        if v1 >= v2:
            raise ValueError(
                "In `list_of_egdes`, vertex that compose an edge must be ordered (cf. `" + str((v1, v2)) + "`)."
            )
        if (v1 < 0) or (number_of_vertex <= v2):
            raise ValueError(
                "In `list_of_egdes`, vertex that compose an edge must be lower than `number_of_vertex` (cf. `("
                + str(v1)
                + ","
                + str(v2)
                + ")` and `"
                + str(number_of_vertex)
                + "`)."
            )

        # Add the edge
        G1[v1 * number_of_vertex + v2, e] = 1

        # Add the symetric edge
        G1[v2 * number_of_vertex + v1, e] = 1

    # Add the symetry of a vertex
    for v in range(number_of_vertex):
        G1[v * number_of_vertex + v, number_of_edges] = 1

    # Get opposite of `G1`.
    G1 = -G1

    # Initialization and definition of parameter `h1` of the solver.
    h1 = -cvxopt.matrix(1.0, (number_of_vertex, number_of_vertex))

    # Solving :
    # Minimize c.T * x
    # Subject to G1 * x = h1
    solver = cvxopt.solvers.sdp(c=c, Gs=[G1], hs=[h1])

    return {
        "theta": solver["x"][number_of_edges],
        # "Z": np.array(sol['ss'][0]),
        # "B": np.array(sol['zs'][0]),
        "solver": solver,
    }
