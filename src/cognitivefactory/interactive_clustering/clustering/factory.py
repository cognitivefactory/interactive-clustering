# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.factory
* Description:  The factory method used to easily initialize a constrained clustering algorithm.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# The needed clustering abstract class methods.
from cognitivefactory.interactive_clustering.clustering.abstract import AbstractConstrainedClustering

# Dependencies needed to constrained clustering implementation.
from cognitivefactory.interactive_clustering.clustering.hierarchical import HierarchicalConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.kmeans import KMeansConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.spectral import SpectralConstrainedClustering


# ==============================================================================
# CLUSTERING FACTORY
# ==============================================================================
def clustering_factory(algorithm: str = "kmeans", **kargs) -> "AbstractConstrainedClustering":
    """
    A factory to create a new instance of a constrained clustering model.

    Args:
        algorithm (str): The identification of model to instantiate. Can be `"hierarchical"` or `"kmeans"` or `"spectral"`. Defaults to `"kmeans"`.
        **kargs (dict): Other parameters that can be used in the instantiation.

    Raises:
        ValueError: if `algorithm` is not implemented.

    Returns:
        AbstractConstraintsClustering: An instance of constrained clustering model.
    """

    # Check that the requested algorithm is implemented.
    if algorithm not in {"hierarchical", "kmeans", "spectral"}:
        raise ValueError("The `algorithm` '" + str(algorithm) + "' is not implemented.")

    # Case of Hierachical Constrained Clustering.
    if algorithm == "hierarchical":

        return HierarchicalConstrainedClustering(**kargs)

    # Case of Spectral Constrained Clustering.
    if algorithm == "spectral":

        return SpectralConstrainedClustering(**kargs)

    # Case of KMeans Constrained Clustering.
    ## if algorithm=="kmeans":
    return KMeansConstrainedClustering(**kargs)
