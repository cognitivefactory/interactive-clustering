# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.constraints.factory
* Description:  The factory method used to easily initialize a constraints manager.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORTS :
# =============================================================================

# Python code typing (mypy).
from typing import List

# The needed clustering abstract class and utils methods.
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager

# Dependencies needed to constraints managing implementation.
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager


# ==============================================================================
# MANAGING FACTORY :
# ==============================================================================
def managing_factory(list_of_data_IDs: List[str], manager: str = "binary", **kargs) -> AbstractConstraintsManager:
    """
    A factory to create a new instance of a constraints manager.

    Args:
        list_of_data_IDs (List[str]): The list of data IDs to manage.
        manager (str, optional): The identification of constraints manager to instantiate. Can be "binary". Defaults to `"binary"`.
        **kargs (dict): Other parameters that can be used in the instantiation.

    Raises:
        ValueError: if `manager` is not implemented.

    Returns:
        AbstractConstraintsManager : An instance of constraints manager.
    """

    # Check that the requested algorithm is implemented.
    if manager != "binary":  # TODO use `not in {"binary"}`.
        raise ValueError("The `manager` '" + str(manager) + "' is not implemented.")

    # Case of Binary Constraints Manager
    ## if manager=="binary":

    return BinaryConstraintsManager(list_of_data_IDs=list_of_data_IDs, **kargs)
