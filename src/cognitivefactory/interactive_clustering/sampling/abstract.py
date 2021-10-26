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

from abc import ABC, abstractmethod  # To define an abstract class.
from typing import Dict, List, Optional, Tuple  # To type Python code (mypy).

from scipy.sparse import csr_matrix  # To handle matrix and vectors.

from cognitivefactory.interactive_clustering.constraints.abstract import (  # To manage constraints.
    AbstractConstraintsManager,
)


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
        constraints_manager: AbstractConstraintsManager,
        nb_to_select: int,
        clustering_result: Optional[Dict[str, int]] = None,
        vectors: Optional[Dict[str, csr_matrix]] = None,
        **kargs,
    ) -> List[Tuple[str, str]]:
        """
        (ABSTRACT METHOD)
        An abstract method that represents the main method used to sample couple of data IDs for constraints annotation.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs.
            nb_to_select (int): The number of couple of data IDs to select.
            clustering_result (Optional[Dict[str,int]], optional): A dictionary that represents the predicted cluster for each data ID. The keys of the dictionary represents the data IDs. If `None`, no clustering result are used during the sampling. Defaults to `None`.
            vectors (Optional[Dict[str, csr_matrix]], optional): vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data. If `None`, no vectors are used during the sampling. Defaults to `None`
            **kargs (dict): Other parameters that can be used in the sampling.

        Raises:
            ValueError: if some parameters are incorrectly set or incompatible.

        Returns:
            List[Tuple[str,str]]: A list of couple of data IDs.
        """
