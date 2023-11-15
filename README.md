# Interactive Clustering

[![ci](https://github.com/cognitivefactory/interactive-clustering/workflows/ci/badge.svg)](https://github.com/cognitivefactory/interactive-clustering/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://cognitivefactory.github.io/interactive-clustering/)
[![pypi version](https://img.shields.io/pypi/v/cognitivefactory-interactive-clustering.svg)](https://pypi.org/project/cognitivefactory-interactive-clustering/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4775251.svg)](https://doi.org/10.5281/zenodo.4775251)

Python package used to apply NLP interactive clustering methods.


## <a name="Description"></a> Quick description

_Interactive clustering_ is a method intended to assist in the design of a training data set.

This iterative process begins with an unlabeled dataset, and it uses a sequence of two substeps :

1. the user defines constraints on data sampled by the computer ;

2. the computer performs data partitioning using a constrained clustering algorithm.

Thus, at each step of the process :

- the user corrects the clustering of the previous steps using constraints, and

- the computer offers a corrected and more relevant data partitioning for the next step.

The process use severals objects :

- a _constraints manager_ : its role is to manage the constraints annotated by the user and to feed back the information deduced (such as the transitivity between constraints or the situation of inconsistency) ;

- a _constraints sampler_ : its role is to select the most relevant data during the annotation of constraints by the user ;

- a _constrained clustering algorithm_ : its role is to partition the data while respecting the constraints provided by the user.

_NB_ :

- This python library does not contain integration into a graphic interface.

- For more details, read the [Documentation](#Documentation) and the articles in the [References](#References) section.


## <a name="Documentation"></a> Documentation

- [Main documentation](https://cognitivefactory.github.io/interactive-clustering/)


## <a name="Installation"></a> Installation

Interactive Clustering requires Python 3.8 or above.

To install with [`pip`](https://github.com/pypa/pip):

```bash
# install package
python3 -m pip install cognitivefactory-interactive-clustering

# install spacy language model dependencies (the one you want, with version "3.4.x")
python3 -m spacy download fr_core_news_md-3.4.0 --direct
```

To install with [`pipx`](https://github.com/pypa/pipx):

```bash
# install pipx
python3 -m pip install --user pipx

# install package
pipx install --python python3 cognitivefactory-interactive-clustering

# install spacy language model dependencies (the one you want, with version "3.4.x")
python3 -m spacy download fr_core_news_md-3.4.0 --direct
```

_NB_ : Other spaCy language models can be downloaded here : [spaCy - Models & Languages](https://spacy.io/usage/models). Use spacy version `"3.4.x"`.


## <a name="Development"></a> Development

To work on this project or contribute to it, please read:

- the [Copier PDM](https://pawamoy.github.io/copier-pdm/) template documentation ;
- the [Contributing](https://cognitivefactory.github.io/interactive-clustering/contributing/) page for environment setup and development help ;
- the [Code of Conduct](https://cognitivefactory.github.io/interactive-clustering/code_of_conduct/) page for contribution rules.


## <a name="References"></a> References

- **Interactive Clustering**:
	- PhD report: `Schild, E. (2024, in press). De l'Importance de Valoriser l'Expertise Humaine dans l'Annotation : Application à la Modélisation de Textes en Intentions à l'aide d'un Clustering Interactif. Université de Lorraine.` ;
	- First presentation: `Schild, E., Durantin, G., Lamirel, J.C., & Miconi, F. (2021). Conception itérative et semi-supervisée d'assistants conversationnels par regroupement interactif des questions. In EGC 2021 - 21èmes Journées Francophones Extraction et Gestion des Connaissances. Edition RNTI. <hal-03133007>.`
	- Theoretical study: `Schild, E., Durantin, G., Lamirel, J., & Miconi, F. (2022). Iterative and Semi-Supervised Design of Chatbots Using Interactive Clustering. International Journal of Data Warehousing and Mining (IJDWM), 18(2), 1-19. http://doi.org/10.4018/IJDWM.298007. <hal-03648041>.`
	- Methodological discussion: `Schild, E., Durantin, G., & Lamirel, J.C. (2021). Concevoir un assistant conversationnel de manière itérative et semi-supervisée avec le clustering interactif. In Atelier - Fouille de Textes - Text Mine 2021 - En conjonction avec EGC 2021. <hal-03133060>.`

- **Constraints and Constrained Clustering**:
	- Constraints in clustering: `Wagstaff, K. et C. Cardie (2000). Clustering with Instance-level Constraints. Proceedings of the Seventeenth International Conference on Machine Learning, 1103–1110.`
	- Survey on Constrained Clustering: `Lampert, T., T.-B.-H. Dao, B. Lafabregue, N. Serrette, G. Forestier, B. Cremilleux, C. Vrain, et P. Gancarski (2018). Constrained distance based clustering for time-series : a comparative and experimental study. Data Mining and Knowledge Discovery 32(6), 1663–1707.`
	- Affinity Propagation:
		- Affinity Propagation Clustering: `Frey, B. J., & Dueck, D. (2007). Clustering by Passing Messages Between Data Points. In Science (Vol. 315, Issue 5814, pp. 972–976). American Association for the Advancement of Science (AAAS). https://doi.org/10.1126/science.1136800`
		- Constrained Affinity Propagation Clustering: `Givoni, I., & Frey, B. J. (2009). Semi-Supervised Affinity Propagation with Instance-Level Constraints. Proceedings of the Twelth International Conference on Artificial Intelligence and Statistics, PMLR 5:161-168`
	- DBScan:
		- DBScan Clustering: `Ester, Martin & Kröger, Peer & Sander, Joerg & Xu, Xiaowei. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. KDD. 96. 226-231`.
		- Constrained DBScan Clustering: `Ruiz, Carlos & Spiliopoulou, Myra & Menasalvas, Ernestina. (2007). C-DBSCAN: Density-Based Clustering with Constraints. 216-223. 10.1007/978-3-540-72530-5_25.`
	- KMeans Clustering:
		- KMeans Clustering: `MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the fifth Berkeley symposium on mathematical statistics and probability 1(14), 281–297.`
		- Constrained _'COP'_ KMeans Clustering: `Wagstaff, K., C. Cardie, S. Rogers, et S. Schroedl (2001). Constrained K-means Clustering with Background Knowledge. International Conference on Machine Learning`
		- Constrained _'MPC'_ KMeans Clustering: `Khan, Md. A., Tamim, I., Ahmed, E., & Awal, M. A. (2012). Multiple Parameter Based Clustering (MPC): Prospective Analysis for Effective Clustering in Wireless Sensor Network (WSN) Using K-Means Algorithm. In Wireless Sensor Network (Vol. 04, Issue 01, pp. 18–24). Scientific Research Publishing, Inc. https://doi.org/10.4236/wsn.2012.41003`
	- Hierarchical Clustering:
		- Hierarchical Clustering: `Murtagh, F. et P. Contreras (2012). Algorithms for hierarchical clustering : An overview. Wiley Interdisc. Rew.: Data Mining and Knowledge Discovery 2, 86–97.`
		- Constrained Hierarchical Clustering: `Davidson, I. et S. S. Ravi (2005). Agglomerative Hierarchical Clustering with Constraints : Theoretical and Empirical Results. Springer, Berlin, Heidelberg 3721, 12.`
	- Spectral Clustering:
		- Spectral Clustering: `Ng, A. Y., M. I. Jordan, et Y.Weiss (2002). On Spectral Clustering: Analysis and an algorithm. In T. G. Dietterich, S. Becker, et Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14. MIT Press.`
		- Constrained _'SPEC'_ Spectral Clustering: `Kamvar, S. D., D. Klein, et C. D. Manning (2003). Spectral Learning. Proceedings of the international joint conference on artificial intelligence, 561–566.`

- **Preprocessing and Vectorization**:
	- _spaCy_: `Honnibal, M. et I. Montani (2017). spaCy 2 : Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.`
		- _spaCy_ language models: `https://spacy.io/usage/models`
	- _NLTK_: `Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.`
		- _NLTK_ _'SnowballStemmer'_: `https://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball`
	- _Scikit-learn_: `Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R.Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, et E. Duchesnay (2011). Scikit-learn : Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.`
		- _Scikit-learn_ _'TfidfVectorizer'_: `https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html`


## <a name="Other links"></a> Other links

- Several comparative studies of Interactive Clustering methodology on NLP datasets: `Schild, E. (2021). cognitivefactory/interactive-clustering-comparative-study. Zenodo. https://doi.org/10.5281/zenodo.5648255`
- A web application designed for NLP data annotation using Interactive Clustering methodology: `Schild, E. (2021). cognitivefactory/interactive-clustering-gui. Zenodo. https://doi.org/10.5281/zenodo.4775270`


## <a name="How to cite"></a> How to cite

`Schild, E. (2021). cognitivefactory/interactive-clustering. Zenodo. https://doi.org/10.5281/zenodo.4775251.`
