# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.sampling.clusters_based
* Description:  Implementation of constraints sampling based on clusters information.
* Author:       Erwan SCHILD
* Created:      04/10/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import random
from typing import Dict, List, Optional, Tuple

from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise_distances

from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager
from cognitivefactory.interactive_clustering.sampling.abstract import AbstractConstraintsSampling


# ==============================================================================
# CLUSTERS BASED CONSTRAINTS SAMPLING
# ==============================================================================
class ClustersBasedConstraintsSampling(AbstractConstraintsSampling):
    """
    This class implements the sampling of data IDs based on clusters information in order to annotate constraints.
    It inherits from `AbstractConstraintsSampling`.

    Example:
        ```python
        # Import.
        from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
        from cognitivefactory.interactive_clustering.sampling.clusters_based import ClustersBasedConstraintsSampling

        # Create an instance of random sampling.
        sampler = ClustersBasedConstraintsSampling(random_seed=1)

        # Define list of data IDs.
        list_of_data_IDs = ["bonjour", "salut", "coucou", "au revoir", "a bientôt",]

        # Define constraints manager.
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
        clusters_restriction: Optional[str] = None,
        distance_restriction: Optional[str] = None,
        without_added_constraints: bool = True,
        without_inferred_constraints: bool = True,
        **kargs,
    ) -> None:
        """
        The constructor for Clusters Based Constraints Sampling class.

        Args:
            random_seed (Optional[int]): The random seed to use to redo the same sampling. Defaults to `None`.
            clusters_restriction (Optional[str]): Restrict the sampling with a cluster constraints. Can impose data IDs to be in `"same_cluster"` or `"different_clusters"`. Defaults to `None`.  # TODO: `"specific_clusters"`
            distance_restriction (Optional[str]): Restrict the sampling with a distance constraints. Can impose data IDs to be `"closest_neighbors"` or `"farthest_neighbors"`. Defaults to `None`.
            without_added_constraints (bool): Option to not sample the already added constraints. Defaults to `True`.
            without_inferred_constraints (bool): Option to not sample the deduced constraints from already added one. Defaults to `True`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Store `self.random_seed`.
        self.random_seed: Optional[int] = random_seed

        # Store clusters restriction.
        if clusters_restriction not in {None, "same_cluster", "different_clusters"}:
            raise ValueError("The `clusters_restriction` '" + str(clusters_restriction) + "' is not implemented.")
        self.clusters_restriction: Optional[str] = clusters_restriction

        # Store distance restriction.
        if distance_restriction not in {None, "closest_neighbors", "farthest_neighbors"}:
            raise ValueError("The `distance_restriction` '" + str(distance_restriction) + "' is not implemented.")
        self.distance_restriction: Optional[str] = distance_restriction

        # Store constraints restrictions.
        if not isinstance(without_added_constraints, bool):
            raise ValueError("The `without_added_constraints` must be boolean")
        self.without_added_constraints: bool = without_added_constraints
        if not isinstance(without_inferred_constraints, bool):
            raise ValueError("The `without_inferred_constraints` must be boolean")
        self.without_inferred_constraints: bool = without_inferred_constraints

    # ==============================================================================
    # MAIN - SAMPLE
    # ==============================================================================
    def sample(
        self,
        constraints_manager: AbstractConstraintsManager,
        nb_to_select: int,
        clustering_result: Optional[Dict[str, int]] = None,
        vectors: Optional[Dict[str, csr_matrix]] = None,
        **kargs,
    ) -> List[Tuple[str, str]]:
        """
        The main method used to sample pairs of data IDs for constraints annotation.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs.
            nb_to_select (int): The number of pairs of data IDs to sample.
            clustering_result (Optional[Dict[str,int]], optional): A dictionary that represents the predicted cluster for each data ID. The keys of the dictionary represents the data IDs. If `None`, no clustering result are used during the sampling. Defaults to `None`.
            vectors (Optional[Dict[str, csr_matrix]], optional): vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data. If `None`, no vectors are used during the sampling. Defaults to `None`
            **kargs (dict): Other parameters that can be used in the sampling.

        Raises:
            ValueError: if some parameters are incorrectly set or incompatible.

        Returns:
            List[Tuple[str,str]]: A list of couple of data IDs.
        """

        ###
        ### GET PARAMETERS
        ###

        # Check `constraints_manager`.
        if not isinstance(constraints_manager, AbstractConstraintsManager):
            raise ValueError("The `constraints_manager` parameter has to be a `AbstractConstraintsManager` type.")
        self.constraints_manager: AbstractConstraintsManager = constraints_manager

        # Check `nb_to_select`.
        if not isinstance(nb_to_select, int) or (nb_to_select < 0):
            raise ValueError("The `nb_to_select` '" + str(nb_to_select) + "' must be greater than or equal to 0.")
        elif nb_to_select == 0:
            return []

        # If `self.cluster_restriction` is set, check `clustering_result` parameters.
        if self.clusters_restriction is not None:
            if not isinstance(clustering_result, dict):
                raise ValueError("The `clustering_result` parameter has to be a `Dict[str, int]` type.")
            self.clustering_result: Dict[str, int] = clustering_result

        # If `self.distance_restriction` is set, check `vectors` parameters.
        if self.distance_restriction is not None:
            if not isinstance(vectors, dict):
                raise ValueError("The `vectors` parameter has to be a `Dict[str, csr_matrix]` type.")
            self.vectors: Dict[str, csr_matrix] = vectors

        ###
        ### DEFINE POSSIBLE PAIRS OF DATA IDS
        ###

        # Initialize possible pairs of data IDs
        list_of_possible_pairs_of_data_IDs: List[Tuple[str, str]] = []

        # Loop over pairs of data IDs.
        for data_ID1 in self.constraints_manager.get_list_of_managed_data_IDs():
            for data_ID2 in self.constraints_manager.get_list_of_managed_data_IDs():
                # Select ordered pairs.
                if data_ID1 >= data_ID2:
                    continue

                # Check clusters restriction.
                if (
                    self.clusters_restriction == "same_cluster"
                    and self.clustering_result[data_ID1] != self.clustering_result[data_ID2]
                ) or (
                    self.clusters_restriction == "different_clusters"
                    and self.clustering_result[data_ID1] == self.clustering_result[data_ID2]
                ):
                    continue

                # Check known constraints.
                if (
                    self.without_added_constraints is True
                    and self.constraints_manager.get_added_constraint(data_ID1=data_ID1, data_ID2=data_ID2) is not None
                ) or (
                    self.without_inferred_constraints is True
                    and self.constraints_manager.get_inferred_constraint(data_ID1=data_ID1, data_ID2=data_ID2)
                    is not None
                ):
                    continue

                # Add the pair of data IDs.
                list_of_possible_pairs_of_data_IDs.append((data_ID1, data_ID2))

        ###
        ### SAMPLING
        ###

        # Precompute pairwise distances.
        if self.distance_restriction is not None:
            # Compute pairwise distances.
            matrix_of_pairwise_distances: csr_matrix = pairwise_distances(
                X=vstack(self.vectors[data_ID] for data_ID in self.constraints_manager.get_list_of_managed_data_IDs()),
                metric="euclidean",  # TODO get different pairwise_distances config in **kargs
            )

            # Format pairwise distances in a dictionary.
            self.dict_of_pairwise_distances: Dict[str, Dict[str, float]] = {
                vector_ID1: {
                    vector_ID2: float(matrix_of_pairwise_distances[i1, i2])
                    for i2, vector_ID2 in enumerate(self.constraints_manager.get_list_of_managed_data_IDs())
                }
                for i1, vector_ID1 in enumerate(self.constraints_manager.get_list_of_managed_data_IDs())
            }

        # Set random seed.
        random.seed(self.random_seed)

        # Case of closest neightbors selection.
        if self.distance_restriction == "closest_neighbors":
            return sorted(
                list_of_possible_pairs_of_data_IDs,
                key=lambda combination: self.dict_of_pairwise_distances[combination[0]][combination[1]],
            )[:nb_to_select]

        # Case of farthest neightbors selection.
        if self.distance_restriction == "farthest_neighbors":
            return sorted(
                list_of_possible_pairs_of_data_IDs,
                key=lambda combination: self.dict_of_pairwise_distances[combination[0]][combination[1]],
                reverse=True,
            )[:nb_to_select]

        # (default) Case of random selection.
        return random.sample(
            list_of_possible_pairs_of_data_IDs, k=min(nb_to_select, len(list_of_possible_pairs_of_data_IDs))
        )
