"""
* Name:         interactive-clustering/src/clustering/affinity_propagation.py
* Description:  Implementation of constrained Affinity Propagation clustering algorithm.
* Author:       David Nicolazo
* Created:      2/03/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import math
import warnings
from itertools import combinations, product
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, vstack  # To handle matrix and vectors.
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances  # To compute distance.
from sklearn.utils import as_float_array, check_random_state

from cognitivefactory.interactive_clustering.clustering.abstract import (  # To use abstract interface.; To sort clusters after computation.
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)
from cognitivefactory.interactive_clustering.constraints.abstract import (  # To manage constraints.
    AbstractConstraintsManager,
)

# ==============================================================================
# AFFINITY PROPAGATION FROM SKLEARN UNDER CONSTRAINTS
# https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/cluster/_affinity_propagation.py
# Modified by David Nicolazo <git@dabsunter.fr> according to https://proceedings.mlr.press/v5/givoni09a.html
# ==============================================================================


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()


def _update_metapoints(S, must_links, cannot_links):
    pass


def _affinity_propagation_constrained(
    S,
    must_links,
    cannot_links,
    preference=None,
    convergence_iter=15,
    max_iter=200,
    damping=0.5,
    copy=True,
    verbose=False,
    return_n_iter=False,
    random_state=None,
    remove_degeneracies=True,
    absolute_must_links=False,
):
    # TODO: Changer le format de la docstring.
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------

    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations.

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    remove_degeneracies : bool, default=True
        Whether to remove degeneracies in the similarity matrix.

    absolute_must_links : bool, default=False
        Whether to use absolute must links implementation.

    Returns
    -------

    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it returns an empty array as
    ``cluster_center_indices`` and ``-1`` as label for each training sample.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]
    n_must_links = len(must_links)
    n_similarities = n_samples + n_must_links

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        _preference = np.array(np.median(S))
    else:
        _preference = np.array(preference)

    if n_samples == 1 or _equal_similarities_and_preferences(S, _preference):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn("All samples have mutually equal similarities. Returning arbitrary cluster center(s).")
        if _preference.flat[0] >= S.flat[n_samples - 1]:
            return (
                (np.arange(n_samples), np.arange(n_samples), 0)
                if return_n_iter
                else (np.arange(n_samples), np.arange(n_samples))
            )
        else:
            return (
                (np.array([0]), np.array([0] * n_samples), 0)
                if return_n_iter
                else (np.array([0]), np.array([0] * n_samples))
            )

    random_state = check_random_state(random_state)

    # Remove degeneracies
    if remove_degeneracies:
        S += (np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100) * random_state.randn(n_samples, n_samples)

    if verbose:
        print("S", S, sep="\n")

    if absolute_must_links:
        Saml = np.zeros((n_must_links, n_must_links))

        for i, j in product(range(n_must_links), range(n_must_links)):
            Saml[i, j] = max(S[k, l] for k, l in product(must_links[i], must_links[j]))

        S = Saml

        n_similarities = n_must_links

        if preference is None:
            _preference = np.array(np.median(S))

    else:
        # Must-link meta-points blocks
        MPS = np.array([[0 if i in Pm else max(S[i, j] for j in Pm) for Pm in must_links] for i in range(n_samples)])

        if verbose:
            print("MPS", MPS, sep="\n")

        #
        S = np.block([[S, MPS], [MPS.T, 2 * np.min(S) * np.ones((n_must_links, n_must_links))]])

    # Place preference on the diagonal of S
    S.flat[:: (n_similarities + 1)] = _preference

    if verbose:
        print("working S", S, sep="\n")

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
    e = np.zeros((n_similarities, convergence_iter))

    ind = np.arange(n_similarities)

    for it in range(max_iter):
        if verbose:
            print(f"Running iteration #{it}...")

        if not absolute_must_links:  # TODO: Cannot link not supported for absolute_must_links mode
            # TODO: Verify Cannot link implementation as its effect is almost inexistant

            # Sp = S + np.sum(Q2, axis=2)
            # np.sum(Q2, axis=2, out=Sp)
            # np.add(S, Sp, Sp)
            Sp[:] = S
            for m in range(n_samples, n_similarities):
                Sp[m, :] += sum(Q2[m, :, n] for n, CLm in enumerate(cannot_links) if m in CLm)

            # if verbose:
            #     print(Q2)

            # Q1 = A + R - Q2
            Q1new = A[:, :, None] + R[:, :, None] - Q2

            # Q2 = - max(0, Q1)
            Q2 = -np.maximum(0, Q1)
            Q1 = Q1new

        # tmp = A + S; compute responsibilities
        np.add(A, Sp, tmp)
        I = np.argmax(tmp, axis=1)
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
        tmp.flat[:: n_similarities + 1] = R.flat[:: n_similarities + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_similarities + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_similarities  # n_samples
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
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
            for l, ml in zip(labels, must_links):
                for i in ml:
                    real_labels[i] = l
            labels = real_labels
        else:
            # Remove meta-points
            labels = labels[:n_samples]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        if verbose:
            warnings.warn(
                "Affinity propagation did not converge, this model " "will not have any cluster centers.",
                ConvergenceWarning,
            )
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


# ==============================================================================
# AFFINITY PROPAGATION CONSTRAINED CLUSTERING
# ==============================================================================


class AffinityPropagationConstrainedClustering(AbstractConstrainedClustering):
    """
    This class will implements the affinity propagation constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    References:
        - Inmar Givoni, Brendan Frey Proceedings of the Twelth International Conference on Artificial Intelligence and Statistics, PMLR 5:161-168, 2009. 

    Examples:
        ...
    """

    def __init__(
        self,
        preference: Union[float, Any] = None,
        max_iteration: int = 200,
        convergence_iter: int = 15,
        random_state=None,
        absolute_must_links: bool = False,
        smart_preference: bool = False,
        smart_preference_factor: float = 2 / 3,
        ensure_nb_clusters: bool = False,
    ) -> None:
        """Initialize the affinity propagation constrained clustering.

        Args:
            preference (Union[float, Any], optional): Preference parameter. Defaults to None.
            max_iteration (int, optional): Nombre maximal d'itérations. Defaults to 200.
            convergence_iter (int, optional): Nombre d'itérations à considérer pour détecter la convergence. Defaults to 15.
            random_state (Any, optional): État aléatoire (seed). Defaults to None.
            absolute_must_links (bool, optional): Utiliser l'implémentations respectant strictement les contraintes must-link. Defaults to False.
            smart_preference (bool, optional): Calculer de manière "intelligente" la paramètre préférence en fonction du nombre de contraintes. Defaults to False.
            smart_preference_factor (float, optional): Paramètre d'ajustement de `smart_preference`. Defaults to 2/3.
            ensure_nb_clusters (bool, optional): Relance un clustering tant que le nombre de cluster ne correspond pas. Defaults to False.

        Raises:
            ValueError: Paramètre invalide.
        """

        self.preference = preference

        # Store 'self.max_iteration`.
        if max_iteration < 1:
            raise ValueError("The `max_iteration` must be greater than or equal to 1.")
        self.max_iteration: int = max_iteration

        self.convergence_iter = convergence_iter

        self.random_state = random_state

        self.absolute_must_links = absolute_must_links

        self.smart_preference = smart_preference

        self.smart_preference_factor = smart_preference_factor

        self.ensure_nb_clusters = ensure_nb_clusters

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================

    def cluster(
        self,
        constraints_manager: AbstractConstraintsManager,
        vectors: Dict[str, csr_matrix],
        nb_clusters: int = 0,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the KMeans model.

        Args:
            constraints_manager (AbstractConstraintsManager): A constraints manager over data IDs that will force clustering to respect some conditions during computation.
            vectors (Dict[str, csr_matrix]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager`. The value of the dictionary represent the vector of each data.
            nb_clusters (int): The number of clusters to compute.  #TODO Set defaults to None with elbow method or other method ?
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

        ###
        ### RUN AFFINITY PROPAGATION CONSTRAINED CLUSTERING
        ###

        # Correspondances ID -> index
        data_ID_to_idx: Dict[str, int] = {v: i for i, v in enumerate(self.list_of_data_IDs)}
        n_sample = len(self.list_of_data_IDs)

        # Calcul de la matrice des similarités entre les points réels
        S: csr_matrix = -pairwise_distances(vstack(self.vectors[data_ID] for data_ID in self.list_of_data_IDs))

        # Récupération des fermetures transitives des contraintes MUST_LINK
        must_link_closures: List[List[str]] = self.constraints_manager.get_connected_components()

        must_links: List[Set[int]] = [{data_ID_to_idx[ID] for ID in closure} for closure in must_link_closures]

        if verbose:
            print("Must-links", *must_links, sep="\n")

        cannot_links: List[Tuple[int, int]] = []

        for i, j in combinations(range(n_sample), 2):
            constraint = self.constraints_manager.get_added_constraint(
                self.list_of_data_IDs[i], self.list_of_data_IDs[j]
            )
            if constraint and constraint[0] == "CANNOT_LINK":
                cannot_links.append((i, j))

        preference = self.preference
        median = np.median(S)

        if self.smart_preference:
            n_must_links = 0
            for data1, data2 in combinations(self.list_of_data_IDs, 2):
                constraint = constraints_manager.get_added_constraint(data1, data2)
                if constraint and constraint[0] == "MUST_LINK":
                    n_must_links += 1

            must_link_completeness = n_must_links / math.comb(len(self.list_of_data_IDs), 2)

            preference = (1 - must_link_completeness) * median * self.smart_preference_factor

        elif not preference:
            preference = median

        def run_ap(preference):
            return _affinity_propagation_constrained(
                S,
                must_links,
                cannot_links,
                preference=preference,
                verbose=verbose,
                max_iter=self.max_iteration,
                convergence_iter=self.convergence_iter,
                random_state=self.random_state,
                absolute_must_links=self.absolute_must_links,
            )

        cluster_center_indices, labels = run_ap(preference)

        # ensure_nb_clusters: Not very conclusive
        if self.ensure_nb_clusters:
            epsilon = 0.005
            max_change = 250

            while (current_nb_clusters := len(cluster_center_indices)) < nb_clusters:
                if current_nb_clusters == 0:
                    change = max_change
                else:
                    change = np.clip(nb_clusters - current_nb_clusters, -max_change, max_change)
                preference += change * -median * epsilon
                if verbose:
                    print(
                        "found",
                        current_nb_clusters,
                        "clusters /",
                        nb_clusters,
                        "wanted, run again with preference:",
                        preference,
                    )
                cluster_center_indices, labels = run_ap(preference)

        self.dict_of_predicted_clusters = rename_clusters_by_order(
            {self.list_of_data_IDs[i]: l for i, l in enumerate(labels)}
        )

        self.number_of_clusters = len(cluster_center_indices)

        if verbose:
            print("dict of predicted clusters:", self.dict_of_predicted_clusters)

        return self.dict_of_predicted_clusters
        # return {i: l for i,l in enumerate(labels)}
