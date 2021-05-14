# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/utils/test_vectorization.py
* Description:  Unittests for the `utils.vectorization` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Needed library to apply tests.
import pytest

# Dependencies needed to handle matrix.
from numpy import ndarray
from scipy.sparse import csr_matrix

# Modules/Classes/Methods to test.
from cognitivefactory.interactive_clustering.utils.vectorization import vectorize


# ==============================================================================
# test_vectorize_for_unimplemented_vectorizer
# ==============================================================================
def test_vectorize_for_unimplemented_vectorizer():
    """
    Test that the `utils.vectorization.vectorize` raises `ValueError` for unimplemented vectorizer.
    """

    # Check a unimplemented vectorizer.
    with pytest.raises(ValueError, match="`vectorizer_type`"):

        vectorize(
            dict_of_texts={
                "0": "comment signaler une perte de carte de paiement",
                "1": "quelle est la procedure pour chercher une carte de credit avalee",
                "2": "ma carte visa a un plafond de paiment trop bas puis je l augmenter",
            },
            vectorizer_type="unimplemented",
        )


# ==============================================================================
# test_vectorize_for_tfidf_vectorizer
# ==============================================================================
def test_vectorize_for_tfidf_vectorizer():
    """
    Test that the `utils.vectorization.vectorize` works for TFIDF vectorizer.
    """

    # Check a TFIDF vectorizer.
    dict_of_vectors = vectorize(
        dict_of_texts={
            "0": "comment signaler une perte de carte de paiement",
            "1": "quelle est la procedure pour chercher une carte de credit avalee",
            "2": "ma carte visa a un plafond de paiment trop bas puis je l augmenter",
        },
        vectorizer_type="tfidf",
    )

    # Assertions
    assert dict_of_vectors
    assert sorted(dict_of_vectors.keys()) == ["0", "1", "2"]
    assert isinstance(dict_of_vectors["0"], (ndarray, csr_matrix))
    assert isinstance(dict_of_vectors["1"], (ndarray, csr_matrix))
    assert isinstance(dict_of_vectors["2"], (ndarray, csr_matrix))


# ==============================================================================
# test_vectorize_for_spacy_vectorizer
# ==============================================================================
def test_vectorize_for_spacy_vectorizer():
    """
    Test that the `utils.vectorization.vectorize` works for SPACY vectorizer.
    """

    # Check a SPACY vectorizer.
    dict_of_vectors = vectorize(
        dict_of_texts={
            "0": "comment signaler une perte de carte de paiement",
            "1": "quelle est la procedure pour chercher une carte de credit avalee",
            "2": "ma carte visa a un plafond de paiment trop bas puis je l augmenter",
        },
        vectorizer_type="spacy",
    )

    # Assertions
    assert dict_of_vectors
    assert sorted(dict_of_vectors.keys()) == ["0", "1", "2"]
    assert isinstance(dict_of_vectors["0"], (ndarray, csr_matrix))
    assert isinstance(dict_of_vectors["1"], (ndarray, csr_matrix))
    assert isinstance(dict_of_vectors["2"], (ndarray, csr_matrix))
