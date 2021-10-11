# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.constraints.binary
* Description:  Implementation of binary constraints manager.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from typing import Dict, List, Optional, Set, Tuple  # To type Python code (mypy).

import networkx as nx  # To create graph.

from cognitivefactory.interactive_clustering.constraints.abstract import (  # To use abstract interface.
    AbstractConstraintsManager,
)


# ==============================================================================
# BINARY CONSTRAINTS MANAGER
# ==============================================================================
class BinaryConstraintsManager(AbstractConstraintsManager):
    """
    This class implements the binary constraints mangement.
    It inherits from `AbstractConstraintsManager`, and it takes into account the strong transitivity of constraints.

    References:
        - Binary constraints in clustering: `Wagstaff, K. et C. Cardie (2000). Clustering with Instance-level Constraints. Proceedings of the Seventeenth International Conference on Machine Learning, 1103–1110.`

    Examples:
        ```python
        # Import.
        from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager

        # Create an instance of binary constraints manager.
        constraints_manager = BinaryConstraintsManager(list_of_data_IDs=["0", "1", "2", "3", "4"])

        # Add new data ID.
        constraints_manager.add_data_ID(data_ID="99")

        # Get list of data IDs.
        constraints_manager.get_list_of_managed_data_IDs()

        # Delete an existing data ID.
        constraints_manager.delete_data_ID(data_ID="99")

        # Add constraints.
        constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="1", data_ID2="2", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="2, data_ID2="3", constraint_type="CANNOT_LINK")

        # Get added constraint.
        constraints_manager.get_added_constraint(data_ID1="0", data_ID2="1")  # expected ("MUST_LINK", 1.0)
        constraints_manager.get_added_constraint(data_ID1="0", data_ID2="2")  # expected None

        # Get inferred constraint.
        constraints_manager.get_inferred_constraint(data_ID1="0", data_ID2="2")  # expected "MUST_LINK"
        constraints_manager.get_inferred_constraint(data_ID1="0", data_ID2="3")  # expected "CANNOT_LINK"
        constraints_manager.get_inferred_constraint(data_ID1="0", data_ID2="4")  # expected None
        ```
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(self, list_of_data_IDs: List[str], **kargs) -> None:

        """
        The constructor for Binary Constraints Manager class.
        This class use the strong transitivity to infer on constraints, so constraints values are not taken into account.

        Args:
            list_of_data_IDs (List[str]): The list of data IDs to manage.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Raises:
            ValueError: if `list_of_data_IDs` has duplicates.
        """

        # Define `self._allowed_constraint_types`.
        self._allowed_constraint_types: Set[str] = {
            "MUST_LINK",
            "CANNOT_LINK",
        }
        # Define `self._allowed_constraint_value_range`.
        self._allowed_constraint_value_range: Dict[str, float] = {
            "min": 1.0,
            "max": 1.0,
        }

        # If `list_of_data_IDs` has duplicates, raise `ValueError`.
        if len(list_of_data_IDs) != len(set(list_of_data_IDs)):
            raise ValueError("There is duplicates in `list_of_data_IDs`.")

        # Store `self.kargs` for binary constraints managing.
        self.kargs = kargs

        # Initialize `self._constraints_dictionary`.
        self._constraints_dictionary: Dict[str, Dict[str, Optional[Tuple[str, float]]]] = {
            data_ID1: {
                data_ID2: (
                    ("MUST_LINK", 1.0)
                    if (data_ID1 == data_ID2)
                    else None  # Unknwon constraints if `data_ID1` != `data_ID2`.
                )
                for data_ID2 in list_of_data_IDs
            }
            for data_ID1 in list_of_data_IDs
        }

        # Define `self._constraints_transitivity`.
        self._generate_constraints_transitivity()

    # ==============================================================================
    # DATA_ID MANAGEMENT - ADDITION
    # ==============================================================================
    def add_data_ID(
        self,
        data_ID: str,
    ) -> bool:

        """
        The main method used to add a new data ID to manage.

        Args:
            data_ID (str): The data ID to manage.

        Raises:
            ValueError: if `data_ID` is already managed.

        Returns:
            bool: `True` if the addition is done.
        """

        # If `data_ID` is in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID` `'" + str(data_ID) + "'` is already managed.")

        # Add `data_ID` to `self._constraints_dictionary.keys()`.
        self._constraints_dictionary[data_ID] = {}

        # Define constraint for `data_ID` and all other data IDs.
        for other_data_ID in self._constraints_dictionary.keys():
            self._constraints_dictionary[data_ID][other_data_ID] = (
                ("MUST_LINK", 1.0)
                if (data_ID == other_data_ID)
                else None  # Unknwon constraints if `data_ID` != `other_data_ID`.
            )
            self._constraints_dictionary[other_data_ID][data_ID] = (
                ("MUST_LINK", 1.0)
                if (data_ID == other_data_ID)
                else None  # Unknwon constraints if `data_ID1` != `other_data_ID`.
            )

        # Regenerate `self._constraints_transitivity`.
        # `Equivalent to `self._generate_constraints_transitivity()`
        self._constraints_transitivity[data_ID] = {
            "MUST_LINK": {data_ID: None},
            "CANNOT_LINK": {},
        }

        # Return `True`.
        return True

    # ==============================================================================
    # DATA_ID MANAGEMENT - DELETION
    # ==============================================================================
    def delete_data_ID(
        self,
        data_ID: str,
    ) -> bool:
        """
        The main method used to delete a data ID to no longer manage.

        Args:
            data_ID (str): The data ID to no longer manage.

        Raises:
            ValueError: if `data_ID` is not managed.

        Returns:
            bool: `True` if the deletion is done.
        """

        # If `data_ID` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID` `'" + str(data_ID) + "'` is not managed.")

        # Remove `data_ID` from `self._constraints_dictionary.keys()`.
        self._constraints_dictionary.pop(data_ID)

        # Remove `data_ID` from all `self._constraints_dictionary[other_data_ID].keys()`.
        for other_data_ID in self._constraints_dictionary.keys():
            self._constraints_dictionary[other_data_ID].pop(data_ID)

        # Regenerate `self._constraints_transitivity`
        self._generate_constraints_transitivity()

        # Return `True`.
        return True

    # ==============================================================================
    # DATA_ID MANAGEMENT - LISTING
    # ==============================================================================
    def get_list_of_managed_data_IDs(
        self,
    ) -> List[str]:
        """
        The main method used to get the list of data IDs that are managed.

        Returns:
            List[str]: The list of data IDs that are managed.
        """

        # Return the possible keys of `self._constraints_dictionary`.
        return list(self._constraints_dictionary.keys())

    # ==============================================================================
    # CONSTRAINTS MANAGEMENT - ADDITION
    # ==============================================================================
    def add_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
        constraint_type: str,
        constraint_value: float = 1.0,
    ) -> bool:
        """
        The main method used to add a constraint between two data IDs.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint addition.
            data_ID2 (str): The second data ID that is concerned for this constraint addition.
            constraint_type (str): The type of the constraint to add. The type have to be `"MUST_LINK"` or `"CANNOT_LINK"`.
            constraint_value (float, optional): The value of the constraint to add. The value have to be in range `[0.0, 1.0]`. Defaults to `1.0`.

        Raises:
            ValueError: if `data_ID1`, `data_ID2`, `constraint_type` are not managed, or if a conflict is detected with constraints inference.

        Returns:
            bool: `True` if the addition is done, `False` is the constraint can't be added.
        """

        # If `data_ID1` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID1 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID1` `'" + str(data_ID1) + "'` is not managed.")

        # If `data_ID2` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID2 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID2` `'" + str(data_ID2) + "'` is not managed.")

        # If the `constraint_type` is not in `self._allowed_constraint_types`, then raises a `ValueError`.
        if constraint_type not in self._allowed_constraint_types:
            raise ValueError(
                "The `constraint_type` `'"
                + str(constraint_type)
                + "'` is not managed. Allowed constraints types are : `"
                + str(self._allowed_constraint_types)
                + "`."
            )

        # Get current added constraint between `data_ID1` and `data_ID2`.
        inferred_constraint: Optional[str] = self.get_inferred_constraint(
            data_ID1=data_ID1,
            data_ID2=data_ID2,
        )

        # Case of conflict with constraints inference.
        if (inferred_constraint is not None) and (inferred_constraint != constraint_type):
            raise ValueError(
                "The `constraint_type` `'"
                + str(constraint_type)
                + "'` is incompatible with the inferred constraint `'"
                + str(inferred_constraint)
                + "'` between data IDs `'"
                + data_ID1
                + "'` and `'"
                + data_ID2
                + "'`."
            )

        # Get current added constraint between `data_ID1` and `data_ID2`.
        added_constraint: Optional[Tuple[str, float]] = self.get_added_constraint(
            data_ID1=data_ID1,
            data_ID2=data_ID2,
        )

        # If the constraint has already be added, ...
        if added_constraint is not None:
            # ... do nothing.
            return True  # `added_constraint[0] == constraint_type`.
        # Otherwise, the constraint has to be added.

        # Add the direct constraint between `data_ID1` and `data_ID2`.
        self._constraints_dictionary[data_ID1][data_ID2] = (constraint_type, 1.0)
        self._constraints_dictionary[data_ID2][data_ID1] = (constraint_type, 1.0)

        # Add the transitivity constraint between `data_ID1` and `data_ID2`.
        self._add_constraint_transitivity(
            data_ID1=data_ID1,
            data_ID2=data_ID2,
            constraint_type=constraint_type,
        )

        return True

    # ==============================================================================
    # CONSTRAINTS MANAGEMENT - DELETION
    # ==============================================================================
    def delete_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
    ) -> bool:
        """
        The main method used to delete a constraint between two data IDs.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint deletion.
            data_ID2 (str): The second data ID that is concerned for this constraint deletion.

        Raises:
            ValueError: if `data_ID1` or `data_ID2` are not managed.

        Returns:
            bool: `True` if the deletion is done, `False` if the constraint can't be deleted.
        """

        # If `data_ID1` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID1 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID1` `'" + str(data_ID1) + "'` is not managed.")

        # If `data_ID2` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID2 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID2` `'" + str(data_ID2) + "'` is not managed.")

        # Delete the constraint between `data_ID1` and `data_ID2`.
        self._constraints_dictionary[data_ID1][data_ID2] = None
        self._constraints_dictionary[data_ID2][data_ID1] = None

        # Regenerate `self._constraints_transitivity`.
        self._generate_constraints_transitivity()

        # Return `True`
        return True

    # ==============================================================================
    # CONSTRAINTS MANAGEMENT - GETTER
    # ==============================================================================
    def get_added_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
    ) -> Optional[Tuple[str, float]]:
        """
        The main method used to get the constraint added between the two data IDs.
        Do not take into account the constraints transitivity, just look at constraints that are explicitly added.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint.
            data_ID2 (str): The second data ID that is concerned for this constraint.

        Raises:
            ValueError: if `data_ID1` or `data_ID2` are not managed.

        Returns:
            Optional[Tuple[str, float]]: `None` if no constraint, `(constraint_type, constraint_value)` otherwise.
        """

        # If `data_ID1` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID1 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID1` `'" + str(data_ID1) + "'` is not managed.")

        # If `data_ID2` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID2 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID2` `'" + str(data_ID2) + "'` is not managed.")

        # Retrun the current added constraint type and value.
        return self._constraints_dictionary[data_ID1][data_ID2]

    # ==============================================================================
    # CONSTRAINTS EXPLORATION - GETTER
    # ==============================================================================
    def get_inferred_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
        threshold: float = 1.0,
    ) -> Optional[str]:
        """
        The main method used to check if the constraint inferred by transitivity between the two data IDs.
        The transitivity is taken into account, and the `threshold` parameter is used to evaluate the impact of constraints transitivity.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint.
            data_ID2 (str): The second data ID that is concerned for this constraint.
            threshold (float, optional): The threshold used to evaluate the impact of constraints transitivity link. Defaults to `1.0`.

        Raises:
            ValueError: if `data_ID1`, `data_ID2` or `threshold` are not managed.

        Returns:
            Optional[str]: The type of the inferred constraint. The type can be `None`, `"MUST_LINK"` or `"CANNOT_LINK"`.
        """

        # If `data_ID1` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID1 not in self._constraints_transitivity.keys():
            raise ValueError("The `data_ID1` `'" + str(data_ID1) + "'` is not managed.")

        # If `data_ID2` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID2 not in self._constraints_transitivity.keys():
            raise ValueError("The `data_ID2` `'" + str(data_ID2) + "'` is not managed.")

        # Case of `"MUST_LINK"`.
        if data_ID1 in self._constraints_transitivity[data_ID2]["MUST_LINK"].keys():
            return "MUST_LINK"

        # Case of `"CANNOT_LINK"`.
        if data_ID1 in self._constraints_transitivity[data_ID2]["CANNOT_LINK"].keys():
            return "CANNOT_LINK"

        # Case of `None`.
        return None

    # ==============================================================================
    # CONSTRAINTS EXPLORATION - LIST OF COMPONENTS GETTER
    # ==============================================================================
    def get_connected_components(
        self,
        threshold: float = 1.0,
    ) -> List[List[str]]:
        """
        The main method used to get the possible lists of data IDs that are linked by a `"MUST_LINK"` constraints.
        Each list forms a component of the constraints transitivity graph, and it forms a partition of the managed data IDs.
        The transitivity is taken into account, and the `threshold` parameters is used if constraints values are used in the constraints transitivity.

        Args:
            threshold (float, optional): The threshold used to define the transitivity link. Defaults to `1.0`.

        Returns:
            List[List[int]]: The list of lists of data IDs that represent a component of the constraints transitivity graph.
        """

        # Initialize the list of connected components.
        list_of_connected_components: List[List[str]] = []

        # For each data ID...
        for data_ID in self._constraints_transitivity.keys():

            # ... get the list of `"MUST_LINK"` data IDs linked by transitivity with `data_ID` ...
            connected_component_of_a_data_ID = list(self._constraints_transitivity[data_ID]["MUST_LINK"].keys())

            # ... and if the connected component is not already get...
            if connected_component_of_a_data_ID not in list_of_connected_components:
                # ... then add it to the list of connected components.
                list_of_connected_components.append(connected_component_of_a_data_ID)

        # Return the list of connected components.
        return list_of_connected_components

    # ==============================================================================
    # CONSTRAINTS EXPLORATION - CHECK COMPLETUDE OF CONSTRAINTS
    # ==============================================================================
    def check_completude_of_constraints(
        self,
        threshold: float = 1.0,
    ) -> bool:
        """
        The main method used to check if all possible constraints are known (not necessarily annotated because of the transitivity).
        The transitivity is taken into account, and the `threshold` parameters is used if constraints values are used in the constraints transitivity.

        Args:
            threshold (float, optional): The threshold used to define the transitivity link. Defaults to `1.0`.

        Returns:
            bool: Return `True` if all constraints are known, `False` otherwise.
        """

        # For each data ID...
        for data_ID in self._constraints_transitivity.keys():

            # ... if some data IDs are not linked by transitivity to this `data_ID` with a `"MUST_LINK"` or `"CANNOT_LINK"` constraints...
            if (
                len(self._constraints_transitivity[data_ID]["MUST_LINK"].keys())
                + len(self._constraints_transitivity[data_ID]["CANNOT_LINK"].keys())
            ) != len(self._constraints_transitivity.keys()):

                # ... then return `False`.
                return False

        # Otherwise, return `True`.
        return True

    # ==============================================================================
    # CONSTRAINTS EXPLORATION - GET MIN AND MAX NUMBER OF CLUSTERS
    # ==============================================================================
    def get_min_and_max_number_of_clusters(
        self,
        threshold: float = 1.0,
    ) -> Tuple[int, int]:
        """
        The main method used to get determine, for a clustering model that would not violate any constraints, the range of the possible clusters number.
        Minimum number of cluster is estimated by the coloration of the `"CANNOT_LINK"` constraints graph.
        Maximum number of cluster is defined by the number of `"MUST_LINK"` connected components.
        The transitivity is taken into account, and the `threshold` parameters is used if constraints values are used in the constraints transitivity.

        Args:
            threshold (float, optional): The threshold used to define the transitivity link. Defaults to `1.0`.

        Returns:
            Tuple[int,int]: The minimum and the maximum possible clusters numbers (for a clustering model that would not violate any constraints).
        """

        # Get the `"MUST_LINK"` connected components.
        list_of_connected_components: List[List[str]] = self.get_connected_components()

        ###
        ### 1. Estimation of minimum clusters number.
        ###

        # Get connected component ids.
        list_of_connected_component_ids: List[str] = [component[0] for component in list_of_connected_components]

        # Keep only components that have more that one `"CANNOT_LINK"` constraints.
        list_of_linked_connected_components_ids: List[str] = [
            component_id
            for component_id in list_of_connected_component_ids
            if len(self._constraints_transitivity[component_id]["CANNOT_LINK"].keys()) > 1  # noqa: WPS507
        ]

        # Get the `"CANNOT_LINK"` constraints.
        list_of_cannot_link_constraints: List[Tuple[int, int]] = [
            (i1, i2)
            for i1, data_ID1 in enumerate(list_of_linked_connected_components_ids)
            for i2, data_ID2 in enumerate(list_of_linked_connected_components_ids)
            if (i1 < i2)
            and (  # To get the complement, get all possible link that are not a `"CANNOT_LINK"`.
                data_ID2 in self._constraints_transitivity[data_ID1]["CANNOT_LINK"].keys()
            )
        ]

        # Create a networkx graph.
        cannot_link_graph: nx.Graph = nx.Graph()
        cannot_link_graph.add_nodes_from(list_of_connected_component_ids)  # Add components id as nodes in the graph.
        cannot_link_graph.add_edges_from(
            list_of_cannot_link_constraints
        )  # Add cannot link constraints as edges in the graph.

        # Estimate the minimum clusters number by trying to colorate the `"CANNOT_LINK"` constraints graph.
        # The lower bound has to be greater than 2.
        estimation_of_minimum_clusters_number: int = max(
            2,
            1
            + min(
                max(nx.coloring.greedy_color(cannot_link_graph, strategy="largest_first").values()),
                max(nx.coloring.greedy_color(cannot_link_graph, strategy="smallest_last").values()),
                max(nx.coloring.greedy_color(cannot_link_graph, strategy="random_sequential").values()),
                max(nx.coloring.greedy_color(cannot_link_graph, strategy="random_sequential").values()),
                max(nx.coloring.greedy_color(cannot_link_graph, strategy="random_sequential").values()),
            ),
        )

        ###
        ### 2. Computation of maximum clusters number.
        ###

        # Determine the maximum clusters number with the number of `"MUST_LINK"` connected components.
        maximum_clusters_number: int = len(list_of_connected_components)

        # Return minimum and maximum.
        return (estimation_of_minimum_clusters_number, maximum_clusters_number)

    # ==============================================================================
    # CONSTRAINTS TRANSITIVITY MANAGEMENT - GENERATE CONSTRAINTS TRANSITIVITY GRAPH
    # ==============================================================================
    def _generate_constraints_transitivity(
        self,
    ) -> None:
        """
        Generate `self._constraints_transitivity`, a constraints dictionary that takes into account the transitivity of constraints.
        Suppose there is no inconsistency in `self._constraints_dictionary`.
        It uses `Dict[str, None]` to simulate ordered sets.
        """

        self._constraints_transitivity: Dict[str, Dict[str, Dict[str, None]]] = {
            data_ID: {
                "MUST_LINK": {data_ID: None},  # Initialize MUST_LINK clusters constraints.
                "CANNOT_LINK": {},  # Initialize CANNOT_LINK clusters constraints.
            }
            for data_ID in self._constraints_dictionary.keys()
        }

        for data_ID1 in self._constraints_transitivity.keys():
            for data_ID2 in self._constraints_transitivity.keys():

                # Optimization : `self._constraints_dictionary` is symetric, so one pass is enough.
                if data_ID1 > data_ID2:
                    continue

                # Get the constraint between `data_ID1` and `data_ID2`.
                constraint = self._constraints_dictionary[data_ID1][data_ID2]

                # Add the constraint transitivity if the constraint is not `None`.
                if constraint is not None:
                    self._add_constraint_transitivity(
                        data_ID1=data_ID1,
                        data_ID2=data_ID2,
                        constraint_type=constraint[0],
                    )

    # ==============================================================================
    # CONSTRAINTS TRANSITIVITY MANAGEMENT - ADD CONSTRAINT TRANSITIVITY
    # ==============================================================================
    def _add_constraint_transitivity(
        self,
        data_ID1: str,
        data_ID2: str,
        constraint_type: str,
    ) -> bool:
        """
        Add constraint transitivity in `self._constraints_transitivity` between `data_ID1` and `data_ID2` for constraint type `constraint_type`.
        Suppose there is no inconsistency in `self._constraints_dictionary`.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint transitivity addition.
            data_ID2 (str): The second data ID that is concerned for this constraint transitivity addition.
            constraint_type (str): The type of the constraint to add. The type have to be `"MUST_LINK"` or `"CANNOT_LINK"`.

        Returns:
            bool: `True` when the transitivity addition is done.
        """

        ###
        ### Case 1 : `constraint_type` is `"MUST_LINK"`.
        ###
        if constraint_type == "MUST_LINK":

            # Define new common set of `"MUST_LINK"` data IDs,
            # by merging the sets of `"MUST_LINK"` data IDs for `data_ID1` and `data_ID2`.
            new_MUST_LINK_common_set: Dict[str, None] = {
                **self._constraints_transitivity[data_ID1]["MUST_LINK"],
                **self._constraints_transitivity[data_ID2]["MUST_LINK"],
            }

            # Define new common set of `"CANNOT_LINK"` data IDs,
            # by merging the sets of `"CANNOT_LINK"` data IDs for `data_ID1` and `data_ID2`.
            new_CANNOT_LINK_common_set: Dict[str, None] = {
                **self._constraints_transitivity[data_ID1]["CANNOT_LINK"],
                **self._constraints_transitivity[data_ID2]["CANNOT_LINK"],
            }

            # For each data that are now similar to `data_ID1` and `data_ID2`...
            for data_ID_ML in new_MUST_LINK_common_set.keys():
                # ... affect the new set of `"MUST_LINK"` constraints...
                self._constraints_transitivity[data_ID_ML]["MUST_LINK"] = new_MUST_LINK_common_set
                # ... and affect the new set of `"CANNOT_LINK"` constraints.
                self._constraints_transitivity[data_ID_ML]["CANNOT_LINK"] = new_CANNOT_LINK_common_set

            # For each data that are now different to `data_ID1` and `data_ID2`...
            for data_ID_CL in new_CANNOT_LINK_common_set.keys():
                # ... affect the new set of `"CANNOT_LINK"` constraints.
                self._constraints_transitivity[data_ID_CL]["CANNOT_LINK"] = {
                    **self._constraints_transitivity[data_ID_CL]["CANNOT_LINK"],
                    **new_MUST_LINK_common_set,
                }

        ###
        ### Case 2 : `constraint_type` is `"CANNOT_LINK"`.
        ###
        else:  # if constraint_type == "CANNOT_LINK":

            # Define new common set of `"CANNOT_LINK"` data IDs for data IDs that are similar to `data_ID1`.
            new_CANNOT_LINK_set_for_data_ID1: Dict[str, None] = {
                **self._constraints_transitivity[data_ID1]["CANNOT_LINK"],
                **self._constraints_transitivity[data_ID2]["MUST_LINK"],
            }

            # Define new common set of `"CANNOT_LINK"` data IDs for data IDs that are similar to `data_ID2`.
            new_CANNOT_LINK_set_for_data_ID2: Dict[str, None] = {
                **self._constraints_transitivity[data_ID2]["CANNOT_LINK"],
                **self._constraints_transitivity[data_ID1]["MUST_LINK"],
            }

            # For each data that are similar to `data_ID1`...
            for data_ID_like_data_ID1 in self._constraints_transitivity[data_ID1]["MUST_LINK"].keys():
                # ... affect the new list of `"CANNOT_LINK"` constraints.
                self._constraints_transitivity[data_ID_like_data_ID1]["CANNOT_LINK"] = new_CANNOT_LINK_set_for_data_ID1

            # For each data that are similar to `data_ID2`...
            for data_ID_like_data_ID2 in self._constraints_transitivity[data_ID2]["MUST_LINK"].keys():
                # ... affect the new list of `"CANNOT_LINK"` constraints.
                self._constraints_transitivity[data_ID_like_data_ID2]["CANNOT_LINK"] = new_CANNOT_LINK_set_for_data_ID2

        # Return `True`
        return True

    # ==============================================================================
    # CONSTRAINTS CONFLICT - GET INVOLVED DATA IDS IN A CONFLICT
    # ==============================================================================
    def get_list_of_involved_data_IDs_in_a_constraint_conflict(
        self,
        data_ID1: str,
        data_ID2: str,
        constraint_type: str,
    ) -> Optional[List[str]]:
        """
        Get all data IDs involved in a constraints conflict.

        Args:
            data_ID1 (str): The first data ID involved in the constraint_conflit.
            data_ID2 (str): The second data ID involved in the constraint_conflit.
            constraint_type (str): The constraint that create a conflict. The constraints can be `"MUST_LINK"` or `"CANNOT_LINK"`.

        Raises:
            ValueError: if `data_ID1`, `data_ID2`, `constraint_type` are not managed.

        Returns:
            Optional[List[str]]: The list of data IDs that are involved in the conflict. It matches data IDs from connected components of `data_ID1` and `data_ID2`.
        """

        # If `data_ID1` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID1 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID1` `'" + str(data_ID1) + "'` is not managed.")

        # If `data_ID2` is not in the data IDs that are currently managed, then raises a `ValueError`.
        if data_ID2 not in self._constraints_dictionary.keys():
            raise ValueError("The `data_ID2` `'" + str(data_ID2) + "'` is not managed.")

        # If the `constraint_conflict` is not in `self._allowed_constraint_types`, then raises a `ValueError`.
        if constraint_type not in self._allowed_constraint_types:
            raise ValueError(
                "The `constraint_type` `'"
                + str(constraint_type)
                + "'` is not managed. Allowed constraints types are : `"
                + str(self._allowed_constraint_types)
                + "`."
            )

        # Case of conflict (after trying to add a constraint different from the inferred constraint).
        if self.get_inferred_constraint(
            data_ID1, data_ID2
        ) is not None and constraint_type != self.get_inferred_constraint(data_ID1, data_ID2):
            return [
                data_ID
                for connected_component in self.get_connected_components()  # Get involved components.
                for data_ID in connected_component  # Get data IDs from these components.
                if data_ID1 in connected_component or data_ID2 in connected_component
            ]

        # Case of no conflict.
        return None
