"""
* Name:         interactive-clustering/src/clustering/affinity_propagation.py
* Description:  Implementation of constrained Affinity Propagation clustering algorithm.
* Author:       David NICOLAZO, Esther LENOTRE, Marc TRUTT
* Created:      02/03/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import warnings
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from cognitivefactory.interactive_clustering.clustering.abstract import (
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager

# ==============================================================================
# AFFINITY PROPAGATION CONSTRAINED CLUSTERING
# ==============================================================================


class AffinityPropagationConstrainedClustering(AbstractConstrainedClustering):
    """
    This class will implements the Affinity Propagation constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    References:
        - Affinity Propagation Clustering: `Frey, B. J., & Dueck, D. (2007). Clustering by Passing Messages Between Data Points. In Science (Vol. 315, Issue 5814, pp. 972–976). American Association for the Advancement of Science (AAAS). https://doi.org/10.1126/science.1136800`
        - Constrained Affinity Propagation Clustering: `Givoni, I., & Frey, B. J. (2009). Semi-Supervised Affinity Propagation with Instance-Level Constraints. Proceedings of the Twelth International Conference on Artificial Intelligence and Statistics, PMLR 5:161-168`

    Example:
        ```python
        # Import.
        from scipy.sparse import csr_matrix
        from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
        from cognitivefactory.interactive_clustering.clustering.affinity_propagation import AffinityPropagationConstrainedClustering

        # Create an instance of affinity propagation clustering.
        clustering_model = AffinityPropagationConstrainedClustering(
            random_seed=1,
        )

        # Define vectors.
        # NB : use cognitivefactory.interactive_clustering.utils to preprocess and vectorize texts.
        vectors = {
            "0": csr_matrix([1.00, 0.00, 0.00, 0.00]),
            "1": csr_matrix([0.95, 0.02, 0.02, 0.01]),
            "2": csr_matrix([0.98, 0.00, 0.02, 0.00]),
            "3": csr_matrix([0.99, 0.00, 0.01, 0.00]),
            "4": csr_matrix([0.60, 0.17, 0.16, 0.07]),
            "5": csr_matrix([0.60, 0.16, 0.17, 0.07]),
            "6": csr_matrix([0.01, 0.01, 0.01, 0.97]),
            "7": csr_matrix([0.00, 0.01, 0.00, 0.99]),
            "8": csr_matrix([0.00, 0.00, 0.00, 1.00]),
        }

        # Define constraints manager.
        constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(vectors.keys()))
        constraints_manager.add_constraint(data_ID1="0", data_ID2="1", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="2", data_ID2="3", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="4", data_ID2="5", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="7", data_ID2="8", constraint_type="MUST_LINK")
        constraints_manager.add_constraint(data_ID1="0", data_ID2="4", constraint_type="CANNOT_LINK")
        constraints_manager.add_constraint(data_ID1="2", data_ID2="4", constraint_type="CANNOT_LINK")
        constraints_manager.add_constraint(data_ID1="4", data_ID2="7", constraint_type="CANNOT_LINK")

        # Run clustering.
        dict_of_predicted_clusters = clustering_model.cluster(
            constraints_manager=constraints_manager,
            vectors=vectors,
            ####nb_clusters=None,
        )

        # Print results.
        print("Expected results", ";", {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 2, "7": 2, "8": 2,})  # TODO:
        print("Computed results", ":", dict_of_predicted_clusters)
        ```

    Warns:
        FutureWarning: `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.
    """

    def __init__(
        self,
        max_iteration: int = 150,
        convergence_iteration: int = 10,
        random_seed: Optional[int] = None,
        absolute_must_links: bool = True,
        **kargs,
    ) -> None:
        """
        The constructor for the Affinity Propagation constrained clustering.

        Args:
            max_iteration (int, optional): The maximum number of iteration for convergence. Defaults to `150`.
            convergence_iteration (int, optional): The number of iterations with no change to consider a convergence. Default to `15`.
            absolute_must_links (bool, optional): the option to strictly respect `"MUST_LINK"` type constraints. Defaults to ``True`.
            random_seed (Optional[int], optional): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Warns:
            FutureWarning: `clustering.affinity_propagation.AffinityPropagationConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Deprecation warnings
        warnings.warn(
            "`clustering.affinity_propagation.AffinityPropagationConstrainedClustering` is still in development and is not fully tested : it is not ready for production use.",
            FutureWarning,  # DeprecationWarning
            stacklevel=2,
        )

        # Store 'self.max_iteration`.
        if max_iteration < 1:
            raise ValueError("The `max_iteration` must be greater than or equal to 1.")
        self.max_iteration: int = max_iteration

        # Store 'self.convergence_iteration`.
        if convergence_iteration < 1:
            raise ValueError("The `convergence_iteration` must be greater than or equal to 1.")
        self.convergence_iteration: int = convergence_iteration

        # Store 'self.absolute_must_links`.
        self.absolute_must_links: bool = absolute_must_links

        # Store 'self.random_seed`.
        self.random_seed: Optional[int] = random_seed

        # Store `self.kargs` for kmeans clustering.
        self.kargs = kargs

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters: Optional[Dict[str, int]] = None

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================

    def cluster(
        self,
        constraints_manager: AbstractConstraintsManager,
        vectors: Dict[str, csr_matrix],
        nb_clusters: Optional[int] = None,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the KMeans model.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs that will force clustering to respect some conditions during computation.
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data.
            nb_clusters (Optional[int]): The number of clusters to compute. Here `None`.
            verbose (bool, optional): Enable verbose output. Defaults to `False`.
            **kargs (dict): Other parameters that can be used in the clustering.

        Raises:
            ValueError: if `vectors` and `constraints_manager` are incompatible, or if some parameters are incorrectly set.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """

        ###
        ### GET PARAMETERS
        ###

        # Store `self.constraints_manager` and `self.list_of_data_IDs`.
        if not isinstance(constraints_manager, AbstractConstraintsManager):
            raise ValueError("The `constraints_manager` parameter has to be a `AbstractConstraintsManager` type.")
        self.constraints_manager: AbstractConstraintsManager = constraints_manager
        self.list_of_data_IDs: List[str] = self.constraints_manager.get_list_of_managed_data_IDs()

        # Store `self.vectors`.
        if not isinstance(vectors, dict):
            raise ValueError("The `vectors` parameter has to be a `dict` type.")
        self.vectors: Dict[str, csr_matrix] = vectors

        # Store `self.nb_clusters`.
        if nb_clusters is not None:
            raise ValueError("The `nb_clusters` should be 'None' for Affinity Propagataion clustering.")
        self.nb_clusters: Optional[int] = None

        ###
        ### RUN AFFINITY PROPAGATION CONSTRAINED CLUSTERING
        ###

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters = None

        # Correspondances ID -> index
        data_ID_to_idx: Dict[str, int] = {v: i for i, v in enumerate(self.list_of_data_IDs)}
        n_sample: int = len(self.list_of_data_IDs)

        # Compute similarity between data points.
        S: csr_matrix = -pairwise_distances(vstack(self.vectors[data_ID] for data_ID in self.list_of_data_IDs))

        # Get connected components (closures of MUST_LINK contraints).
        must_link_closures: List[List[str]] = self.constraints_manager.get_connected_components()
        must_links: List[List[int]] = [[data_ID_to_idx[ID] for ID in closure] for closure in must_link_closures]

        # Get annotated CANNOT_LINK contraints.
        cannot_links: List[Tuple[int, int]] = []
        for data_ID_i1, data_ID_j1 in combinations(range(n_sample), 2):
            constraint = self.constraints_manager.get_added_constraint(
                self.list_of_data_IDs[data_ID_i1], self.list_of_data_IDs[data_ID_j1]
            )
            if constraint and constraint[0] == "CANNOT_LINK":
                cannot_links.append((data_ID_i1, data_ID_j1))

        # Run constrained affinity propagation.
        cluster_labels: List[int] = _affinity_propagation_constrained(
            S,
            must_links=must_links,
            cannot_links=cannot_links,
            absolute_must_links=self.absolute_must_links,
            max_iteration=self.max_iteration,
            convergence_iteration=self.convergence_iteration,
            random_seed=self.random_seed,
            verbose=verbose,
        )

        # Rename cluster IDs by order.
        self.dict_of_predicted_clusters = rename_clusters_by_order(
            {self.list_of_data_IDs[i]: l for i, l in enumerate(cluster_labels)}
        )

        return self.dict_of_predicted_clusters


# ==============================================================================
# AFFINITY PROPAGATION FROM SKLEARN UNDER CONSTRAINTS
# ==============================================================================


def _equal_similarities_and_preferences(S, preference) -> bool:
    def _all_equal_preferences() -> bool:  # noqa: WPS430 (nested function)
        return bool(np.all(preference == preference.flat[0]))

    def _all_equal_similarities() -> bool:  # noqa: WPS430 (nested function)
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return bool(np.all(S[mask].flat == S[mask].flat[0]))

    return _all_equal_preferences() and _all_equal_similarities()


def _affinity_propagation_constrained(
    S: csr_matrix,
    must_links: List[List[int]],
    cannot_links: List[Tuple[int, int]],
    absolute_must_links: bool = True,
    max_iteration: int = 150,
    convergence_iteration: int = 10,
    damping: float = 0.5,
    remove_degeneracies: bool = True,
    random_seed: Optional[int] = None,
    verbose: bool = False,
):
    """
    Perform Affinity Propagation Clustering of data.

    Forked from https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/cluster/_affinity_propagation.py
    Modified by David NICOLAZO <git@dabsunter.fr> according to https://proceedings.mlr.press/v5/givoni09a.html

    Args:
        S (csr_matrix): Matrix of similarities between points.
        must_links (List[List[int]]): The list of MUST_LINK closures, i.e. list of list of data linked by MUST_LINK constraints.
        cannot_links (List[Tuple[int, int]]): The list of data linked by CANNOT_LINK constraints
        absolute_must_links (bool, optional): The option to use absolute must links implementation. Defaults to `True`.
        max_iteration (int, optional): The maximum number of iteration for convergence. Defaults to `150`.
        convergence_iteration (int, optional): The number of iterations with no change to consider a convergence. Default to `15`.
        damping (float, optional): Damping factor between 0.5 and 1. Defaults to `0.5`.
        remove_degeneracies (bool, optional): The option to remove degeneracies in the similarity matrix. Defaults to `True`.
        random_seed (Optional[int], optional): The random seed to use to redo the same clustering. Defaults to `None`.
        verbose (bool, optional): Enable verbose output. Defaults to `False`.

    Returns:
        List[int]: The list of data cluster labels.
    """

    n_samples: int = S.shape[0]
    n_must_links: int = len(must_links)
    n_similarities: int = n_samples + n_must_links

    # Define the preference by the median of the similarity matrix.
    preference = np.array(np.median(S))

    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        # It makes no sense to run the algorithm in this case, so return 1 or n_samples clusters, depending on preferences
        # TODO: warnings.warn("All samples have mutually equal similarities. Returning arbitrary cluster center(s).")
        if preference.flat[0] >= S.flat[n_samples - 1]:
            return np.arange(n_samples)
        return [0 for _ in range(n_samples)]

    # Fix random seed.
    random_state = check_random_state(random_seed)

    # Remove degeneracies
    if remove_degeneracies:
        S += (np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100) * random_state.randn(n_samples, n_samples)

    # Use only meta-points to force data from MUST_LINK closure to be in the same cluster.
    if absolute_must_links:
        Saml = np.zeros((n_must_links, n_must_links))

        for data_ID_i1, data_ID_j1 in product(range(n_must_links), range(n_must_links)):
            Saml[data_ID_i1, data_ID_j1] = max(
                S[k, l] for k, l in product(must_links[data_ID_i1], must_links[data_ID_j1])
            )

        S = Saml

        n_similarities = n_must_links

        preference = np.array(np.median(S))

    # Use data and meta-points.
    else:
        MPS = np.array(
            [
                [0 if data_ID_i2 in Pm else max(S[data_ID_i2, data_ID_j2] for data_ID_j2 in Pm) for Pm in must_links]
                for data_ID_i2 in range(n_samples)
            ]
        )

        # Update similarity matrix.
        S = np.block([[S, MPS], [MPS.T, 2 * np.min(S) * np.ones((n_must_links, n_must_links))]])

    # Place preference on the diagonal of S
    S.flat[:: (n_similarities + 1)] = preference  # noqa: WPS362 (subscript slice)

    if absolute_must_links:
        Sp = S

    A = np.zeros((n_similarities, n_similarities))
    R = np.zeros((n_similarities, n_similarities))  # Initialize messages
    if not absolute_must_links:
        # E-I: CL constraints
        Q1 = np.zeros((n_similarities, n_similarities, len(cannot_links)))  # qj (m, mn) [m,j,n]
        Q2 = np.zeros((n_similarities, n_similarities, len(cannot_links)))  # qj (mn, m) [m,j,n]
        Sp = np.zeros((n_similarities, n_similarities))  # Ŝ

    # Intermediate results
    tmp = np.zeros((n_similarities, n_similarities))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_similarities, convergence_iteration))

    ind = np.arange(n_similarities)

    for it in range(max_iteration):
        if not absolute_must_links:  # TODO: Cannot link not supported for absolute_must_links mode
            # TODO: Verify Cannot link implementation as its effect is almost inexistant

            # Sp = S + np.sum(Q2, axis=2)
            # np.sum(Q2, axis=2, out=Sp)
            # np.add(S, Sp, Sp)
            Sp[:] = S  # noqa: WPS362 (subscript slice)
            for m in range(n_samples, n_similarities):
                Sp[m, :] += sum(Q2[m, :, n] for n, CLm in enumerate(cannot_links) if m in CLm)

            # Q1 = A + R - Q2
            Q1new = A[:, :, None] + R[:, :, None] - Q2

            # Q2 = - max(0, Q1)
            Q2 = -np.maximum(0, Q1)
            Q1 = Q1new

        # tmp = A + S; compute responsibilities
        np.add(A, Sp, tmp)
        I = np.argmax(tmp, axis=1)  # noqa: E741 (ambiguous variable)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(Sp, Y[:, None], tmp)
        tmp[ind, I] = Sp[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[:: n_similarities + 1] = R.flat[:: n_similarities + 1]  # noqa: WPS362 (subscript slice)

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_similarities + 1] = dA  # noqa: WPS362 (subscript slice)

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iteration] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iteration:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iteration) + (se == 0)) != n_similarities  # n_samples
            if (not unconverged and (K > 0)) or (it == max_iteration):
                never_converged = False
                break
    else:
        never_converged = True

    I = np.flatnonzero(E)  # noqa: E741 (ambiguous variable)
    K = I.size  # Identify exemplars

    if K > 0 and not never_converged:
        c = np.argmax(Sp[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(Sp[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        if absolute_must_links:
            # Assign labels to meta-points members
            real_labels = [-1 for _ in range(n_samples)]
            for label, ml in zip(labels, must_links):
                for i in ml:
                    real_labels[i] = label
            labels = real_labels
        else:
            # Remove meta-points
            labels = labels[:n_samples]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        # TODO: if verbose:  # pragma: no cover
        # TODO: warnings.warn("Affinity propagation did not converge, this model " "will not have any cluster centers.", ConvergenceWarning)
        labels = [-1 for _ in range(n_samples)]
        cluster_centers_indices = []

    return labels
