# Interactive Clustering

[![ci](https://github.com/cognitivefactory/interactive-clustering/workflows/ci/badge.svg)](https://github.com/cognitivefactory/interactive-clustering/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://cognitivefactory.github.io/interactive-clustering/)
[![pypi version](https://img.shields.io/pypi/v/interactive-clustering.svg)](https://pypi.org/project/interactive-clustering/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/interactive-clustering/community)

Python package used to apply NLP interactive clustering methods.

## Quick description

_Interactive clustering_ is a method intended to assist in the design of a training data set.

This iterative process begins with an unlabeled dataset, and it uses a sequence of two substeps :
1. the user defines constraints on the data ;
2. the machine performs data partitioning using a constrained clustering algorithm.

Thus, at each step of the process :
- the user corrects the clustering of the previous steps using constraints, and
- the machine offers a corrected and more relevant data partitioning for the next step.

The process use severals objects :
- a _constraints manager_ : its role is to manage the constraints annotated by the user and to feed back the information deduced (such as the transitivity between constraints or the situation of inconsistency) ;
- a _constraints sampler_ : its role is to select the most relevant data during the annotation of constraints by the user ;
- a _constrained clustering algorithm_ : its role is to partition the data while respecting the constraints provided by the user.

_NB_ :
- This python library does not contain integration into a graphic interface.
- For more details, read the articles in the [References](#references) section.

## Requirements

Interactive Clustering requires Python 3.6 or above.

<details>
<summary>To install Python 3.6, I recommend using <a href="https://github.com/pyenv/pyenv"><code>pyenv</code></a>.</summary>

```bash
# install pyenv
git clone https://github.com/pyenv/pyenv ~/.pyenv

# setup pyenv (you should also put these three lines in .bashrc or similar)
export PATH="${HOME}/.pyenv/bin:${PATH}"
export PYENV_ROOT="${HOME}/.pyenv"
eval "$(pyenv init -)"

# install Python 3.6
pyenv install 3.6.12

# make it available globally
pyenv global system 3.6.12
```
</details>

## Installation

With `pip`:
```bash
python3.6 -m pip install cognitivefactory-interactive-clustering
```

With [`pipx`](https://github.com/pipxproject/pipx):
```bash
python3.6 -m pip install --user pipx

pipx install --python python3.6 cognitivefactory-interactive-clustering
```

## References

- **Interactive Clustering**:
    - Theory and Implementation: `Erwan Schild, Gautier Durantin, Jean-Charles Lamirel, Florian Miconi. Conception itérative et semi-supervisée d'assistants conversationnels par regroupement interactif des questions. EGC 2021 - 21èmes Journées Francophones Extraction et Gestion des Connaissances, Association EGC, Jan 2021, Montpellier / Virtual, France. ⟨hal-03133007⟩`
    - Methodological instructions: `Erwan Schild, Gautier Durantin, Jean-Charles Lamirel. Concevoir un assistant conversationnel de manière itérative et semi-supervisée avec le clustering interactif. Atelier - Fouille de Textes - Text Mine 2021 - En conjonction avec EGC 2021, Association EGC, Jan 2021, Montpellier / Virtual, France. ⟨hal-03133060⟩`

- **Constraints and Constrained Clustering**:
    - Constraints in clustering: `Wagstaff, K. et C. Cardie (2000). Clustering with Instance-level Constraints. Proceedings of the Seventeenth International Conference on Machine Learning, 1103–1110.`
    - Survey on Constrained Clustering: `Lampert, T., T.-B.-H. Dao, B. Lafabregue, N. Serrette, G. Forestier, B. Cremilleux, C. Vrain, et P. Gancarski (2018). Constrained distance based clustering for time-series : a comparative and experimental study. Data Mining and Knowledge Discovery 32(6), 1663–1707.`
    - KMeans Clustering:
        - KMeans Clustering: `MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the fifth Berkeley symposium on mathematical statistics and probability 1(14), 281–297.`
        - Constrained _'COP'_ KMeans Clustering: `Wagstaff, K., C. Cardie, S. Rogers, et S. Schroedl (2001). Constrained K-means Clustering with Background Knowledge. International Conference on Machine Learning`
    - Hierarchical Clustering:
        - Hierarchical Clustering: `Murtagh, F. et P. Contreras (2012). Algorithms for hierarchical clustering : An overview. Wiley Interdisc. Rew.: Data Mining and Knowledge Discovery 2, 86–97.`
        - Constrained Hierarchical Clustering: `Davidson, I. et S. S. Ravi (2005). Agglomerative Hierarchical Clustering with Constraints : Theoretical and Empirical Results. Springer, Berlin, Heidelberg 3721, 12.`
    - Spectral Clustering:
        - Spectral Clustering: `Ng, A. Y., M. I. Jordan, et Y.Weiss (2002). On Spectral Clustering: Analysis and an algorithm. In T. G. Dietterich, S. Becker, et Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14. MIT Press.`
        - Constrained _'SPEC'_ Spectral Clustering: `Kamvar, S. D., D. Klein, et C. D. Manning (2003). Spectral Learning. Proceedings of the international joint conference on artificial intelligence, 561–566.`

- **Preprocessing and Vectorization**:
    - _spaCy_: `Honnibal, M. et I. Montani (2017). spaCy 2 : Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.`
    - _spaCy_ language model `fr_core_news_sm`: `https://spacy.io/usage/models`
    - _NLTK_: `Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.`
    - _NLTK_ _'SnowballStemmer'_: `https://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball`
    - _Scikit-learn_: `Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R.Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, et E. Duchesnay (2011). Scikit-learn : Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.`
    - _Scikit-learn_ _'TfidfVectorizer'_: `https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html`
