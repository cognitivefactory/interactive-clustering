# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering
* Description:  Python package used to apply NLP interactive clustering methods.
* Author:       Erwan SCHILD
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)

Three modules are available:

- `constraints`: it provides a constraints manager, that stores annotated constraints on data and gives some feedback on information deduced (such as the transitivity between constraints or the situation of inconsistency). See [interactive_clustering/constraints](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/constraints/) documentation ;
- `sampling`: it provides several constraints sampling algorithm, that selecte relevant contraints to annotate by an expert. See [interactive_clustering/sampling](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/sampling/) documentation ;
- `clustering`: it provides several constrained clustering algorithms, that partition the data according to annotated constraints. See [interactive_clustering/clustering](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/clustering/) documentation ;
- `utils`: it provides several basic functionnalities, like data preprocessing and data vectorization. See [interactive_clustering/utils](https://cognitivefactory.github.io/interactive-clustering/reference/cognitivefactory/interactive_clustering/utils/) documentation.
"""

from typing import List

__all__: List[str] = []  # noqa: WPS410 (only __variable__ we use)
