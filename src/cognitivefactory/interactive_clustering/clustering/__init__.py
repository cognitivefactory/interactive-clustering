# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering
* Description:  Constrained clustering module of the Interactive Clustering package.
* Author:       Erwan SCHILD
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)

This module provides several constrained clustering algorithms, that partition the data according to annotated constraints :

- `abstract`: an abstract class that defines constrained clustering algorithms functionnalities. See [interactive_clustering/clustering/abstract](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/abstract/) documentation ;
- `factory`: a factory to easily instantiate constrained clustering algorithm object. See [interactive_clustering/clustering/factory](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/factory/) documentation ;
- `kmeans`: a constrained clustering algorithm implementation that uses COP-KMeans. See [interactive_clustering/clustering/kmeans](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/kmeans/) documentation ;
- `hierarchical`: a constrained clustering algorithm implementation that uses constrained hierarchical clustering. See [interactive_clustering/clustering/hierarchical](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/hierarchical/) documentation ;
- `spectral`: a constrained clustering algorithm implementation that uses constrained spectral clustering. See [interactive_clustering/clustering/spectral](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/spectral/) documentation ;
- `affinity_propagation`: a constrained clustering algorithm implementation that uses constrained affinity propagation clustering (_not production ready !_). See [interactive_clustering/clustering/affinity_propagation](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/affinity_propagation/) documentation ;
- `dbscan`: a constrained clustering algorithm implementation that uses C-DBScan (_not production ready !_). See [interactive_clustering/clustering/dbscan](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/dbscan/) documentation ;
- `mpckmeans`: a constrained clustering algorithm implementation that uses MPC-KMeans (_not production ready !_). See [interactive_clustering/clustering/mpckmeans](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/mpckmeans/) documentation.
"""
