# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.sampling.abstract
* Description:  The abstract class used to define constraints sampling algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Dependency needed to define an abstract class.
from abc import ABC, abstractmethod

# Python code typing (mypy).
from typing import Dict, List, Optional, Tuple, Union

# Dependencies needed to handle matrix.
from numpy import ndarray
from scipy.sparse import csr_matrix

# Dependency needed to manage constraints.
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager


# ==============================================================================
# ABSTRACT CONSTRAINTS SAMPLING
# ==============================================================================
class AbstractConstraintsSampling(ABC):
    """
    Abstract class that is used to define constraints sampling algorithms.
    The main inherited method is `sample`.
    """

    # ==============================================================================
    # ABSTRACT METHOD - SAMPLE
    # ==============================================================================
    @abstractmethod
    def sample(
        self,
        list_of_data_IDs: List[str],
        nb_to_select: int,
        constraints_manager: Optional[AbstractConstraintsManager] = None,
        clustering_result: Optional[Dict[str, int]] = None,
        vectors: Optional[Dict[str, Union[ndarray, csr_matrix]]] = None,
        **kargs,
    ) -> List[Tuple[str, str]]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to sample couple of data IDs for constraints annotation.

        Args:
            list_of_data_IDs (List[str]): The list of possible data IDs that can be selected.
            nb_to_select (int): The number of couple of data IDs to select.
            constraints_manager (Optional[AbstractConstraintsManager], optional): A constraints manager over data IDs. The list of data IDs managed by `constraints_manager` has to refer to `list_of_data_IDs`. If `None`, no constraint are used during the sampling. Defaults to `None`.
            clustering_result (Optional[Dict[str,int]], optional): A dictionary that represents the predicted cluster for each data ID. The keys of the dictionary has to refer to `list_of_data_IDs`. If `None`, no clustering result are used during the sampling. Defaults to `None`.
            vectors (Optional[Dict[str,Union[ndarray,csr_matrix]]], optional): The representation of data vectors. The keys of the dictionary has to refer to `list_of_data_IDs`. The value of the dictionary represent the vector of each data. Vectors can be dense (`numpy.ndarray`) or sparse (`scipy.sparse.csr_matrix`). If `None`, no vectors are used during the sampling. Defaults to `None`
            **kargs (dict): Other parameters that can be used in the sampling.

        Raises:
            ValueError: if some parameters are incorrectly set or incompatible.

        Returns:
            List[Tuple[str,str]]: A list of couple of data IDs.
        """
