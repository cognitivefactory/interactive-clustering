# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.sampling.farhest_in_same_cluster
* Description:  Implementation of farhest constraints sampling algorithms for data IDs in same cluster.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Dependency needed to shuffle data and set random seed.
import random

# Python code typing (mypy).
from typing import Dict, List, Optional, Tuple, Union

# Dependencies needed to handle matrix.
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

# Dependency needed to compute distance between two data points.
from sklearn.metrics import pairwise_distances

# Dependency needed to manage constraints.
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager

# The needed clustering abstract class and utilities methods.
from cognitivefactory.interactive_clustering.sampling.abstract import AbstractConstraintsSampling
from cognitivefactory.interactive_clustering.utils import checking


# ==============================================================================
# FARHEST IN SAME CLUSTER SAMPLING
# ==============================================================================
class FarhestInSameClusterConstraintsSampling(AbstractConstraintsSampling):
    """
    This class implements the selection of sampling of farhest data IDs from same cluster.
    It inherits from `AbstractConstraintsSampling`.
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
        The constructor for Farhest In Same Cluster Constraints Sampling class.

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
        list_of_data_IDs: List[str],
        nb_to_select: int,
        constraints_manager: Optional[AbstractConstraintsManager] = None,
        clustering_result: Optional[Dict[str, int]] = None,
        vectors: Optional[Dict[str, Union[ndarray, csr_matrix]]] = None,
        **kargs,
    ) -> List[Tuple[str, str]]:
        """
        The main method used to sample couple of data IDs for constraints annotation.

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
            List[Tuple[str,str]]: A list of couple of data IDs
        """

        ###
        ### GET PARAMETERS
        ###

        # Check `list_of_data_IDs`.
        if not isinstance(list_of_data_IDs, list) or not all(isinstance(element, str) for element in list_of_data_IDs):
            raise ValueError("The `list_of_data_IDs` parameter has to be a `list` type.")
        list_of_data_IDs = sorted(list_of_data_IDs)

        # Check `nb_to_select`.
        if not isinstance(nb_to_select, int) or (nb_to_select < 0):
            raise ValueError("The `nb_to_select` '" + str(nb_to_select) + "' must be greater than or equal to 0.")
        elif nb_to_select == 0:
            return []

        # Check `constraints_manager`.
        verified_constraints_manager: AbstractConstraintsManager = checking.check_constraints_manager(
            list_of_data_IDs=list_of_data_IDs,
            constraints_manager=constraints_manager,
        )

        # Check `clustering_result`.
        verified_clustering_result: Dict[str, int] = checking.check_clustering_result(
            list_of_data_IDs=list_of_data_IDs,
            clustering_result=clustering_result,
        )

        # Check `vectors`.
        verified_vectors: Dict[str, Union[ndarray, csr_matrix]] = checking.check_vectors(
            list_of_data_IDs=list_of_data_IDs,
            vectors=vectors,
        )

        ###
        ### RANDOM SELECTION
        ###

        # Get the list of possible combinations.
        list_of_possible_combinations: List[Tuple[str, str]] = [
            (data_ID1, data_ID2)
            for data_ID1 in list_of_data_IDs
            for data_ID2 in list_of_data_IDs
            if (
                # Filter on ordered data IDs.
                data_ID1
                < data_ID2
            )
            and (
                # Filter on unkown data IDs constraints.
                verified_constraints_manager.get_inferred_constraint(
                    data_ID1=data_ID1,
                    data_ID2=data_ID2,
                )
                is None
            )
            and (
                # Filter on data IDs from same cluster.
                verified_clustering_result[data_ID1]
                == verified_clustering_result[data_ID2]
            )
        ]

        # Shuffle the list of possible combinations.
        random.seed(self.random_seed)
        random.shuffle(list_of_possible_combinations)

        # Sorted list of possible combinations by distance (max to min).
        list_of_possible_combinations = sorted(
            list_of_possible_combinations,
            key=lambda combination: pairwise_distances(
                X=verified_vectors[combination[0]],
                Y=verified_vectors[combination[1]],
                metric="euclidean",  # TODO get different pairwise_distances config in **kargs
            )[0][0].astype(np.float64),
            reverse=True,
        )

        # Subset indices.
        list_of_selected_combinations: List[Tuple[str, str]] = list_of_possible_combinations[:nb_to_select]

        return list_of_selected_combinations
