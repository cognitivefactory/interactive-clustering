# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.sampling.factory
* Description:  The factory method used to easily initialize a constraints sampling algorithm.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from cognitivefactory.interactive_clustering.sampling.abstract import (  # To use abstract interface.
    AbstractConstraintsSampling,
)
from cognitivefactory.interactive_clustering.sampling.closest_in_different_clusters import (  # To use closest in different clusters sampling.
    ClosestInDifferentClustersConstraintsSampling,
)
from cognitivefactory.interactive_clustering.sampling.farhest_in_same_cluster import (  # To use farhest in same cluster sampling.
    FarhestInSameClusterConstraintsSampling,
)
from cognitivefactory.interactive_clustering.sampling.random import RandomConstraintsSampling  # To use random sampling.
from cognitivefactory.interactive_clustering.sampling.random_in_same_cluster import (  # To use farhest in same cluster sampling.
    RandomInSameClusterConstraintsSampling,
)


# ==============================================================================
# SAMPLING FACTORY
# ==============================================================================
def sampling_factory(algorithm: str, **kargs) -> "AbstractConstraintsSampling":
    """
    A factory to create a new instance of a constraints sampling model.

    Args:
        algorithm (str): The identification of model to instantiate. Can be `"random"` or `"random_in_same_cluster"` or `"farhest_in_same_cluster"` or `"closest_in_different_clusters"`. Defaults to `"random"`
        **kargs (dict): Other parameters that can be used in the instantiation.

    Raises:
        ValueError: if `algorithm` is not implemented.

    Returns:
        AbstractConstraintsSampling: An instance of constraints sampling model.

    Examples:
        ```python
        # Import.
        from cognitivefactory.interactive_clustering.sampling.factory import sampling_factory

        # Create an instance of random sampler.
        sampler = sampling_factory(
            algorithm="random",
        )
        ```
    """

    # Check that the requested algorithm is implemented.
    if algorithm not in {
        "random",
        "random_in_same_cluster",
        "farhest_in_same_cluster",
        "closest_in_different_clusters",
    }:
        raise ValueError("The `algorithm` '" + str(algorithm) + "' is not implemented.")

    # Case of Random In Same Cluster Constraints Sampling.
    if algorithm == "random_in_same_cluster":
        return RandomInSameClusterConstraintsSampling(**kargs)

    # Case of Farhest In Same Cluster Constraints Sampling.
    if algorithm == "farhest_in_same_cluster":
        return FarhestInSameClusterConstraintsSampling(**kargs)

    # Case of Closest In Different Clusters Constraints Sampling.
    if algorithm == "closest_in_different_clusters":
        return ClosestInDifferentClustersConstraintsSampling(**kargs)

    # Case of Random Constraints Sampling.
    ##if algorithm == "random":
    return RandomConstraintsSampling(**kargs)
