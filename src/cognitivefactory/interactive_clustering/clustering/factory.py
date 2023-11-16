# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.factory
* Description:  The factory method used to easily initialize a constrained clustering algorithm.
* Author:       Erwan SCHILD
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

from cognitivefactory.interactive_clustering.clustering.abstract import AbstractConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.affinity_propagation import (
    AffinityPropagationConstrainedClustering,
)
from cognitivefactory.interactive_clustering.clustering.dbscan import DBScanConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.hierarchical import HierarchicalConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.kmeans import KMeansConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.mpckmeans import MPCKMeansConstrainedClustering
from cognitivefactory.interactive_clustering.clustering.spectral import SpectralConstrainedClustering


# ==============================================================================
# CLUSTERING FACTORY
# ==============================================================================
def clustering_factory(algorithm: str = "kmeans", **kargs) -> "AbstractConstrainedClustering":
    """
    A factory to create a new instance of a constrained clustering model.

    Args:
        algorithm (str): The identification of model to instantiate. Can be `"affinity_propagation"`, `"dbscan"`, `"hierarchical"`, `"kmeans"`, `"mpckmeans"` or `"spectral"`. Defaults to `"kmeans"`.
        **kargs (dict): Other parameters that can be used in the instantiation.

    Warns:
        FutureWarning: `clustering.affinity_propagation.AffinityPropagationConstrainedClustering`, `clustering.dbscan.DBScanConstrainedClustering` and `clustering.mpckmeans.MPCKMeansConstrainedClustering` are still in development and are not fully tested : it is not ready for production use.

    Raises:
        ValueError: if `algorithm` is not implemented.

    Returns:
        AbstractConstraintsClustering: An instance of constrained clustering model.

    Example:
        ```python
        # Import.
        from cognitivefactory.interactive_clustering.clustering.factory import clustering_factory

        # Create an instance of kmeans.
        clustering_model = clustering_factory(
            algorithm="kmeans",
            model="COP",
            random_seed=42,
        )
        ```
    """

    # Check that the requested algorithm is implemented.
    if algorithm not in {
        "affinity_propagation",
        "dbscan",
        "hierarchical",
        "kmeans",
        "mpckmeans",
        "spectral",
    }:
        raise ValueError("The `algorithm` '" + str(algorithm) + "' is not implemented.")

    # Initialize
    cluster_object: AbstractConstrainedClustering

    # Case of Affinity Propagation Constrained Clustering.
    if algorithm == "affinity_propagation":
        cluster_object = AffinityPropagationConstrainedClustering(**kargs)

    # Case of DBScan Constrained Clustering.
    elif algorithm == "dbscan":
        cluster_object = DBScanConstrainedClustering(**kargs)

    # Case of Hierachical Constrained Clustering.
    elif algorithm == "hierarchical":
        cluster_object = HierarchicalConstrainedClustering(**kargs)

    # Case of MPC KMmeans Constrained Clustering.
    elif algorithm == "mpckmeans":
        cluster_object = MPCKMeansConstrainedClustering(**kargs)

    # Case of Spectral Constrained Clustering.
    elif algorithm == "spectral":
        cluster_object = SpectralConstrainedClustering(**kargs)

    # Default case of KMeans Constrained Clustering (algorithm=="kmeans":).
    else:
        cluster_object = KMeansConstrainedClustering(**kargs)

    # Return cluster object.
    return cluster_object
