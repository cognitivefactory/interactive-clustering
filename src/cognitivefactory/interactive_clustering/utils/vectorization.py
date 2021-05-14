# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.utils.vectorization
* Description:  Utilities methods to apply NLP vectorization.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Import path management dependencies.
import os

# Python code typing (mypy).
from typing import Dict, Union

# Dependencies needed to handle float and matrix.
import numpy as np
import spacy
from numpy import ndarray
from scipy.sparse import csr_matrix

# Dependencies for vectorization.
from sklearn.feature_extraction.text import TfidfVectorizer


# ==============================================================================
# NLP VECTORIZATION
# ==============================================================================
def vectorize(
    dict_of_texts: Dict[str, str],
    vectorizer_type: str = "tfidf",
    # TODO language (str, optional) : #TODO. Defaults to `"fr"`.
) -> Dict[str, Union[ndarray, csr_matrix]]:
    """
    A method used to vectorize texts.
    Severals vectorizer are available : TFIDF, spaCy language model.

    References:
        - _Scikit-learn_: `Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R.Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, et E. Duchesnay (2011). Scikit-learn : Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.`
        - _Scikit-learn_ _'TfidfVectorizer'_: `https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html`
        - _spaCy_: `Honnibal, M. et I. Montani (2017). spaCy 2 : Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.`
        - _spaCy_ language model `fr_core_news_sm`: `https://spacy.io/usage/models`

    Args:
        dict_of_texts (Dict[str,str]): A dictionary that contains the texts to vectorize.
        vectorizer_type (str, optional): The vectorizer type to use. The type can be `"tfidf"` or `"spacy"`. Defaults to `"tfidf"`.

    Raises:
        ValueError: Raises error if vectorization `type` is not implemented.

    Returns:
        Dict[str, Union[ndarray,csr_matrix]]: A dictionary that contains the computed vectors.
    """

    # Initialize dictionary of vectors.
    dict_of_vectors: Dict[str, Union[ndarray, csr_matrix]] = {}

    ###
    ### Case of TFIDF vectorization.
    ###
    if vectorizer_type == "tfidf":

        # Initialize vectorizer.
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
        )

        # Apply vectorization.
        tfidf_vectorization: csr_matrix = vectorizer.fit_transform(
            [dict_of_texts[data_ID] for data_ID in dict_of_texts.keys()]
        )

        # Format dictionary of vectors to return.
        dict_of_vectors = {
            data_ID: tfidf_vectorization[i].astype(np.float64) for i, data_ID in enumerate(dict_of_texts.keys())
        }

        # Return the dictionary of vectors.
        return dict_of_vectors

    ###
    ### Case of SPACY vectorization.
    ###
    if vectorizer_type == "spacy":

        # Load vectorizer (spaCy language model).
        path_to_model: str = os.path.dirname(os.path.realpath(__file__)) + "/" + "fr_core_news_sm-2.3.0/"
        spacy_nlp = spacy.load(
            name=path_to_model,
            disable=[
                "tagger",  # Not needed
                "parser",  # Not needed
                "ner",  # Not needed
            ],
        )

        # Apply vectorization.
        dict_of_vectors = {
            data_ID: spacy_nlp(text).vector.astype(np.float64) for data_ID, text in dict_of_texts.items()
        }

        # Return the dictionary of vectors.
        return dict_of_vectors

    ###
    ### Other case : Raise a `ValueError`.
    ###
    raise ValueError("The `vectorizer_type` '" + str(vectorizer_type) + "' is not implemented.")
