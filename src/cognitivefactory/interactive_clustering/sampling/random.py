# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.sampling.random
* Description:  Implementation of random constraints sampling algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import random  # To shuffle data and set random seed.
from typing import Dict, List, Optional, Tuple, Union  # To type Python code (mypy).

from numpy import ndarray  # To handle matrix and vectors.
from scipy.sparse import csr_matrix  # To handle matrix and vectors.

from cognitivefactory.interactive_clustering.constraints.abstract import (  # To manage constraints.
    AbstractConstraintsManager,
)
from cognitivefactory.interactive_clustering.sampling.abstract import (  # To use abstract interface.
    AbstractConstraintsSampling,
)


# ==============================================================================
# RANDOM CONSTRAITS SAMPLING
# ==============================================================================
class RandomConstraintsSampling(AbstractConstraintsSampling):
    """
    This class implements the selection of sampling of random data IDs.
    It inherits from `AbstractConstraintsSampling`.

    Examples:
        ```python
        # Import.
        from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
        from cognitivefactory.interactive_clustering.sampling.random import RandomConstraintsSampling

        # Create an instance of random sampling.
        sampler = RandomConstraintsSampling(random_seed=1)

        # Define list of data IDs.
        list_of_data_IDs = ["bonjour", "salut", "coucou", "au revoir", "a bientôt",]

        # Define constraints manager (set it to None for no constraints).
        constraints_manager = BinaryConstraintsManager(
            list_of_data_IDs=list_of_data_IDs,
        )
        constraints_manager.add_constraint(data_ID1="bonjour", data_ID2="salut", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="au revoir", data_ID2="a bientôt", constraint_type="MUST_LINK")

        # Run sampling.
        selection = sampler.sample(
            constraints_manager=constraints_manager,
            nb_to_select=3,
        )

        # Print results.
        print("Expected results", ";", [("au revoir", "bonjour"), ("bonjour", "coucou"), ("a bientôt", "coucou"),])
        print("Computed results", ":", selection)
        ```
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(
        self,
        random_seed: Optional[int] = None,
        **kargs,
    ) -> None:
        """
        The constructor for Random Constraints Sampling class.

        Args:
            random_seed (Optional[int]): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.
        """

        # Store `self.random_seed`.
        self.random_seed: Optional[int] = random_seed

    # ==============================================================================
    # MAIN - SAMPLE
    # ==============================================================================
    def sample(
        self,
        constraints_manager: AbstractConstraintsManager,
        nb_to_select: int,
        clustering_result: Optional[Dict[str, int]] = None,
        vectors: Optional[Dict[str, Union[ndarray, csr_matrix]]] = None,
        **kargs,
    ) -> List[Tuple[str, str]]:
        """
        The main method used to sample couple of data IDs for constraints annotation.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs.
            nb_to_select (int): The number of couple of data IDs to select.
            clustering_result (Optional[Dict[str,int]], optional): A dictionary that represents the predicted cluster for each data ID. The keys of the dictionary represents the data IDs. If `None`, no clustering result are used during the sampling. Defaults to `None`.
            vectors (Optional[Dict[str,Union[ndarray,csr_matrix]]], optional): vectors (Dict[str,Union[ndarray,csr_matrix]]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data. Vectors can be dense (`numpy.ndarray`) or sparse (`scipy.sparse.csr_matrix`). If `None`, no vectors are used during the sampling. Defaults to `None`
            **kargs (dict): Other parameters that can be used in the sampling.

        Raises:
            ValueError: if some parameters are incorrectly set or incompatible.

        Returns:
            List[Tuple[str,str]]: A list of couple of data IDs.
        """

        ###
        ### GET PARAMETERS
        ###

        # Store `self.constraints_manager` and `self.list_of_data_IDs`.
        if not isinstance(constraints_manager, AbstractConstraintsManager):
            raise ValueError("The `constraints_manager` parameter has to be a `AbstractConstraintsManager` type.")
        self.constraints_manager: AbstractConstraintsManager = constraints_manager
        self.list_of_data_IDs: List[str] = self.constraints_manager.get_list_of_managed_data_IDs()

        # Check `nb_to_select`.
        if not isinstance(nb_to_select, int) or (nb_to_select < 0):
            raise ValueError("The `nb_to_select` '" + str(nb_to_select) + "' must be greater than or equal to 0.")
        elif nb_to_select == 0:
            return []

        ###
        ### RANDOM SELECTION
        ###

        # Get the list of possible combinations.
        list_of_possible_combinations: List[Tuple[str, str]] = [
            (data_ID1, data_ID2)
            for data_ID1 in self.list_of_data_IDs
            for data_ID2 in self.list_of_data_IDs
            if (
                # Filter on ordered data IDs.
                data_ID1
                < data_ID2
            )
            and (
                # Filter on unkown data IDs constraints.
                self.constraints_manager.get_inferred_constraint(
                    data_ID1=data_ID1,
                    data_ID2=data_ID2,
                )
                is None
            )
        ]

        # Return a sample the list of possible combinations.
        random.seed(self.random_seed)
        return random.sample(list_of_possible_combinations, k=min(nb_to_select, len(list_of_possible_combinations)))
