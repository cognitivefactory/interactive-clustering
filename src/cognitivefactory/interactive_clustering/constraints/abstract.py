# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.constraints.abstract
* Description:  The abstract class used to define constraints managing algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORTS :
# ==============================================================================

from abc import ABC, abstractmethod  # To define an abstract class.
from typing import List, Optional, Tuple  # To type Python code (mypy).


# ==============================================================================
# ABSTRACT CONSTRAINTS MANAGING
# ==============================================================================
class AbstractConstraintsManager(ABC):
    """
    Abstract class that is used to define constraints manager.
    The main inherited methods are about data IDs management, constraints management and constraints exploration.

    References:
        - Constraints in clustering: `Wagstaff, K. et C. Cardie (2000). Clustering with Instance-level Constraints. Proceedings of the Seventeenth International Conference on Machine Learning, 1103–1110.`
    """

    # ==============================================================================
    # ABSTRACT METHOD - DATA_ID MANAGEMENT
    # ==============================================================================
    @abstractmethod
    def add_data_ID(
        self,
        data_ID: str,
    ) -> bool:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to add a new data ID to manage.

        Args:
            data_ID (str): The data ID to manage.

        Raises:
            ValueError: if `data_ID` is already managed.

        Returns:
            bool: `True` if the addition is done.
        """

    @abstractmethod
    def delete_data_ID(
        self,
        data_ID: str,
    ) -> bool:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to delete a data ID to no longer manage.

        Args:
            data_ID (str): The data ID to no longer manage.

        Raises:
            ValueError: if `data_ID` is not managed.

        Returns:
            bool: `True` if the deletion is done.
        """

    @abstractmethod
    def get_list_of_managed_data_IDs(
        self,
    ) -> List[str]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to get the list of data IDs that are managed.

        Returns:
            List[str]: The list of data IDs that are managed.
        """

    # ==============================================================================
    # ABSTRACT METHOD - CONSTRAINTS MANAGEMENT
    # ==============================================================================
    @abstractmethod
    def add_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
        constraint_type: str,
        constraint_value: float = 1.0,
    ) -> bool:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to add a constraint between two data IDs.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint addition.
            data_ID2 (str): The second data ID that is concerned for this constraint addition.
            constraint_type (str): The type of the constraint to add. The type have to be `"MUST_LINK"` or `"CANNOT_LINK"`.
            constraint_value (float, optional): The value of the constraint to add. The value have to be in range `[0.0, 1.0]`. Defaults to 1.0.

        Raises:
            ValueError: if `data_ID1`, `data_ID2`, `constraint_type` are not managed, or if a conflict is detected with constraints inference.

        Returns:
            bool: `True` if the addition is done, `False` is the constraint can't be added.
        """

    @abstractmethod
    def delete_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
    ) -> bool:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to delete the constraint between two data IDs.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint deletion.
            data_ID2 (str): The second data ID that is concerned for this constraint deletion.

        Raises:
            ValueError: if `data_ID1` or `data_ID2` are not managed.

        Returns:
            bool: `True` if the deletion is done.
        """

    @abstractmethod
    def get_added_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
    ) -> Optional[Tuple[str, float]]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to get the constraint added between the two data IDs.
        Do not take into account the constraints transitivity, just look at constraints that are explicitly added.

        Args:
            data_ID1 (str): The first data ID that is concerned for this constraint.
            data_ID2 (str): The second data ID that is concerned for this constraint.

        Raises:
            ValueError: if `data_ID1` or `data_ID2` are not managed.

        Returns:
            Optional[Tuple[str, float]]: `None` if no constraint, `(constraint_type, constraint_value)` otherwise.
        """

    # ==============================================================================
    # ABSTRACT METHOD - CONSTRAINTS EXPLORATION
    # ==============================================================================
    @abstractmethod
    def get_inferred_constraint(
        self,
        data_ID1: str,
        data_ID2: str,
        threshold: float = 1.0,
    ) -> Optional[str]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to check if the constraint inferred by transitivity between the two data IDs.
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

    @abstractmethod
    def get_connected_components(
        self,
        threshold: float = 1.0,
    ) -> List[List[str]]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to get the possible lists of data IDs that are connected by a `"MUST_LINK"` constraints.
        Each list forms a component of the constraints transitivity graph, and it forms a partition of the managed data IDs.
        The transitivity is taken into account, and the `threshold` parameter is used to evaluate the impact of constraints transitivity.

        Args:
            threshold (float, optional): The threshold used to evaluate the impact of constraints transitivity link. Defaults to `1.0`.

        Raises:
            ValueError: if `threshold` is not managed.

        Returns:
            List[List[int]]: The list of lists of data IDs that represent a component of the constraints transitivity graph.
        """

    @abstractmethod
    def check_completude_of_constraints(
        self,
        threshold: float = 1.0,
    ) -> bool:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to check if all possible constraints are known (not necessarily annotated because of the transitivity).
        The transitivity is taken into account, and the `threshold` parameter is used to evaluate the impact of constraints transitivity.

        Args:
            threshold (float, optional): The threshold used to evaluate the impact of constraints transitivity link. Defaults to `1.0`.

        Raises:
            ValueError: if `threshold` is not managed.

        Returns:
            bool: Return `True` if all constraints are known, `False` otherwise.
        """

    @abstractmethod
    def get_min_and_max_number_of_clusters(
        self,
        threshold: float = 1.0,
    ) -> Tuple[int, int]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to get determine, for a clustering model that would not violate any constraints, the range of the possible clusters number.
        The transitivity is taken into account, and the `threshold` parameter is used to evaluate the impact of constraints transitivity.

        Args:
            threshold (float, optional): The threshold used to evaluate the impact of constraints transitivity link. Defaults to `1.0`.

        Raises:
            ValueError: if `threshold` is not managed.

        Returns:
            Tuple[int,int]: The minimum and the maximum possible clusters numbers (for a clustering model that would not violate any constraints).
        """

    # ==============================================================================
    # ABSTRACT METHOD - CONSTRAINTS CONFLICT
    # ==============================================================================

    @abstractmethod
    def get_list_of_involved_data_IDs_in_a_constraint_conflict(
        self,
        data_ID1: str,
        data_ID2: str,
        constraint_type: str,
    ) -> Optional[List[str]]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to get all data IDs involved in a constraints conflict.

        Args:
            data_ID1 (str): The first data ID involved in the constraint_conflit.
            data_ID2 (str): The second data ID involved in the constraint_conflit.
            constraint_type (str): The constraint that create a conflict. The constraints can be `"MUST_LINK"` or `"CANNOT_LINK"`.

        Raises:
            ValueError: if `data_ID1`, `data_ID2`, `constraint_type` are not managed.

        Returns:
            Optional[List[str]]: The list of data IDs that are involved in the conflict. It matches data IDs from connected components of `data_ID1` and `data_ID2`.
        """
