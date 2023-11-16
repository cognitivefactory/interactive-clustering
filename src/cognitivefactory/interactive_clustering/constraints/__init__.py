# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.constraints
* Description:  Constraints managing module of the Interactive Clustering package.
* Author:       Erwan SCHILD
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)

This module provides a constraints manager, that stores annotated constraints on data and gives some feedback on information deduced (such as the transitivity between constraints or the situation of inconsistency) :

- `abstract`: an abstract class that defines constraints managers functionnalities. See [interactive_clustering/constraints/abstract](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/constraints/abstract/) documentation ;
- `factory`: a factory to easily instantiate constraints manager object. See [interactive_clustering/constraints/factory](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/constraints/factory/) documentation ;
- `binary`: a constraints manager implementation that handles `MUST-LINK` and `CANNOT-LINK` constraints on pairs of data. See [interactive_clustering/constraints/binary](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/constraints/binary/) documentation.
"""
