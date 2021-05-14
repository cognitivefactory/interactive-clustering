# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.utils.checking
* Description:  Utilities methods for checking parameters.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Python code typing (mypy).
from typing import Dict, List, Optional, Union

# Dependencies needed to handle matrix.
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

# Dependency needed to manage constraints.
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# CHECK CONSTRAINTS MANAGER
# ==============================================================================
def check_constraints_manager(
    list_of_data_IDs: List[str],
    constraints_manager: Optional[AbstractConstraintsManager] = None,
) -> AbstractConstraintsManager:
    """
    The main method used to check that constraints manager used in constraints sampling algorithms is valid and compatible.

    Args:
        list_of_data_IDs (List[str]): The list of possible data IDs that can be selected.
        constraints_manager (Optional[AbstractConstraintsManager], optional): A constraints manager over data IDs. The list of data IDs managed by `constraints_manager` has to refer to `list_of_data_IDs`. If `None`, no constraint are used during the sampling. Defaults to `None`.

    Raises:
        ValueError: if some parameters are incorrectly set or incompatible.

    Returns:
        AbstractConstraintsManager: In the case of checks are ok, returns the `constraints_manager` with some adjustments. If `constraints_manager` is `None`, initialize a default `cognitivefactory.constraints_managing.binary.BinaryConstraintsManager` with the same list of data IDs.
    """

    # Check `list_of_data_IDs` type.
    if not isinstance(list_of_data_IDs, list) or not all(isinstance(element, str) for element in list_of_data_IDs):
        raise ValueError("The `list_of_data_IDs` parameter has to be a `list` type.")
    list_of_data_IDs = sorted(list_of_data_IDs)

    # Check `constraints_manager` type.
    if constraints_manager is None:
        constraints_manager = BinaryConstraintsManager(
            list_of_data_IDs=list_of_data_IDs,
        )

    elif not isinstance(constraints_manager, AbstractConstraintsManager):
        raise ValueError(
            "The `constraints_manager` parameter must be `None` or an instance of `cognitivefactory.constraints_managing.abstract.AbstractConstraintsManager`"
        )

    # Check `constraints_manager` data IDs.
    if sorted(constraints_manager.get_list_of_managed_data_IDs()) != list_of_data_IDs:
        raise ValueError("The data IDs of `constraints_manager` parameter has to be in `list_of_data_IDs` parameters.")

    # Return needed values.
    return constraints_manager


# ==============================================================================
# CHECK VECTORS
# ==============================================================================
def check_vectors(
    list_of_data_IDs: List[str],
    vectors: Optional[Dict[str, Union[ndarray, csr_matrix]]],
) -> Dict[str, Union[ndarray, csr_matrix]]:
    """
    The main method used to check that vectors used in constraints sampling algorithms are valid and compatible.

    Args:
        list_of_data_IDs (List[str]): The list of possible data IDs that can be selected.
        vectors (Dict[str,Union[ndarray,csr_matrix]]): The representation of data vectors. The keys of the dictionary has to refer to `list_of_data_IDs`. The value of the dictionary represent the vector of each data. Vectors can be dense (`numpy.ndarray`) or sparse (`scipy.sparse.csr_matrix`).

    Raises:
        ValueError: if some parameters are incorrectly set or incompatible.

    Returns:
        Dict[str,Union[ndarray,csr_matrix]]: In the case of checks are ok, returns the `vectors`. If vectors are 1D `ndarray`, it forces the 2D `ndarray` type with `np.atleast_2d`.
    """

    # Check `list_of_data_IDs` type.
    if not isinstance(list_of_data_IDs, list) or not all(isinstance(element, str) for element in list_of_data_IDs):
        raise ValueError("The `list_of_data_IDs` parameter has to be a `list` type.")
    list_of_data_IDs = sorted(list_of_data_IDs)

    # Check `vectors` type.
    if not isinstance(vectors, dict):
        raise ValueError("The `vectors` parameter has to be a `dict` type.")

    # Check `vectors` data IDs.
    if sorted(vectors.keys()) != list_of_data_IDs:
        raise ValueError("The data IDs of `vectors` parameter has to be in `list_of_data_IDs` parameters.")

    # Check `vectors` type and shape.
    vectors_reference_shape = None
    for data_ID in list_of_data_IDs:

        # Check type is `ndarray` or `csr_matrix`.
        if not isinstance(vectors[data_ID], (ndarray, csr_matrix)):
            raise ValueError(
                "The `vectors` parameter for data ID '"
                + str(data_ID)
                + "' has to be in dense (`numpy.ndarray`) or sparse (`scipy.sparse.csr_matrix`) format."
            )

        # Case of less than 2D `ndarray` : set to 2D `ndarray`.
        if len(vectors[data_ID].shape) < 2:
            vectors[data_ID] = np.atleast_2d(vectors[data_ID])

        # Set reference shape as the shape of the first data ID. Used in next checks.
        if vectors_reference_shape is None:
            vectors_reference_shape = vectors[data_ID].shape

        # Check that vectors are 2D matrix (dense or sparse)
        if len(vectors[data_ID].shape) != 2:
            raise ValueError(
                "The `vectors` parameters for data ID `'"
                + str(data_ID)
                + "'` is not a 2D array. The shape is `"
                + str(vectors[data_ID].shape)
                + "`."
            )
        if vectors[data_ID].shape[0] != 1:
            raise ValueError(
                "The `vectors` parameters for data ID `'"
                + str(data_ID)
                + "'` has multiple row, but has to have only one row. The shape is `"
                + str(vectors[data_ID].shape)
                + "` and must be `(1,vector_dimension)`."
            )
        if vectors[data_ID].shape[1] != vectors_reference_shape[1]:
            raise ValueError(
                "The `vectors` parameters for data ID `'"
                + str(data_ID)
                + "'` has a different vector dimension from the orther data IDs, but all data IDs have to have the same vertor dimension. The shape is `"
                + str(vectors[data_ID].shape)
                + "` and must be `(1,vector_dimension)`."
            )

    # Return needed value.
    return vectors


# ==============================================================================
# CHECK CLUSTERING RESULTS
# ==============================================================================
def check_clustering_result(
    list_of_data_IDs: List[str],
    clustering_result: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """
    The main method used to check that `clustering_result` used in constraints sampling algorithms is valid and compatible.

    Args:
        list_of_data_IDs (List[str]): The list of possible data IDs that can be selected.
        clustering_result (Dict[str,int]): The dictionary that contains the predicted cluster for each data ID.

    Raises:
        ValueError: if some parameters are incorrectly set or incompatible.

    Returns:
        Dict[str,int]: In the case of checks are ok, returns the `clustering_result` with some adjustments.
    """

    # Check `list_of_data_IDs` type.
    if not isinstance(list_of_data_IDs, list) or not all(isinstance(element, str) for element in list_of_data_IDs):
        raise ValueError("The `list_of_data_IDs` parameter has to be a `list` type.")
    list_of_data_IDs = sorted(list_of_data_IDs)

    # Check `clustering_result` type.
    if not isinstance(clustering_result, dict):
        raise ValueError(" The `clustering_result` parameter must be an instance of `dict`.")

    # Check `clustering_result` data IDs.
    if sorted(clustering_result.keys()) != list_of_data_IDs:
        raise ValueError("The data IDs of `clustering_result` parameter has to be in `list_of_data_IDs` parameters.")

    # Check `clustering_result` value type.
    if not all(isinstance(value, int) for _, value in clustering_result.items()):
        raise ValueError("The values of `clustering_result` parameter must be instance `int`.")

    # Return needed values.
    return clustering_result
