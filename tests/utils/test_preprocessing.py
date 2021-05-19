# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/utils/test_preprocessing.py
* Description:  Unittests for the `utils.preprocessing` module.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import pytest

from cognitivefactory.interactive_clustering.utils.preprocessing import preprocess


# ==============================================================================
# test_preprocess_for_uninstalled_spacy_language_model
# ==============================================================================
def test_preprocess_for_uninstalled_spacy_language_model():
    """
    Test that the `utils.preprocessing.preprocess` raises `ValueError` for uninstalled spacy language model.
    """

    # Check a unimplemented vectorizer.
    with pytest.raises(ValueError, match="`spacy_language_model`"):

        preprocess(
            dict_of_texts={
                "0": "Comment signaler une perte de carte de paiement ?",
                "1": "Quelle est la procédure pour chercher une carte de crédit avalée ?",
                "2": "Ma carte Visa a un plafond de paiment trop bas, puis-je l'augmenter ?",
            },
            spacy_language_model="uninstalled",
        )


# ==============================================================================
# test_preprocess_for_installed_spacy_language_model
# ==============================================================================
def test_preprocess_for_installed_spacy_language_model():
    """
    Test that the `utils.preprocessing.preprocess` works for an installed spacy language model.
    """

    # Check simple preprocessing.
    dict_of_preprocessed_texts = preprocess(
        dict_of_texts={
            "0": "Hello. How are you ??",
            "1": "Hello, how old are you ?",
            "2": "Hello ! Where do you live ?",
        },
        spacy_language_model="en_core_web_sm",
    )

    # Assertions
    assert dict_of_preprocessed_texts


# ==============================================================================
# test_preprocess_for_simple_preprocessing
# ==============================================================================
def test_preprocess_for_simple_preprocessing():
    """
    Test that the `utils.preprocessing.preprocess` works for simple preprocessing.
    """

    # Check simple preprocessing.
    dict_of_preprocessed_texts = preprocess(
        dict_of_texts={
            "0": "Comment signaler une perte de carte de paiement ?",
            "1": "Quelle est la procédure pour chercher une carte de crédit avalée ?",
            "2": "Ma carte Visa a un plafond de paiment trop bas, puis-je l'augmenter ?",
        },
    )

    # Assertions
    assert dict_of_preprocessed_texts
    assert sorted(dict_of_preprocessed_texts.keys()) == ["0", "1", "2"]
    assert dict_of_preprocessed_texts["0"] == "comment signaler une perte de carte de paiement"
    assert dict_of_preprocessed_texts["1"] == "quelle est la procedure pour chercher une carte de credit avalee"
    assert dict_of_preprocessed_texts["2"] == "ma carte visa a un plafond de paiment trop bas puis je l augmenter"


# ==============================================================================
# test_preprocess_for_stopwords_deletion
# ==============================================================================
def test_preprocess_for_stopwords_deletion():
    """
    Test that the `utils.preprocessing.preprocess` works for stopwords deletion.
    """

    # Check stopwords deletion.
    dict_of_preprocessed_texts = preprocess(
        dict_of_texts={
            "0": "Comment signaler une perte de carte de paiement ?",
            "1": "Quelle est la procédure pour chercher une carte de crédit avalée ?",
            "2": "Ma carte Visa a un plafond de paiment trop bas, puis-je l'augmenter ?",
        },
        apply_stopwords_deletion=True,
    )

    # Assertions
    assert dict_of_preprocessed_texts
    assert sorted(dict_of_preprocessed_texts.keys()) == ["0", "1", "2"]
    assert dict_of_preprocessed_texts["0"] == "signaler perte carte paiement"
    assert dict_of_preprocessed_texts["1"] == "procedure chercher carte credit avalee"
    assert dict_of_preprocessed_texts["2"] == "carte visa plafond paiment l augmenter"


# ==============================================================================
# test_preprocess_for_parsing_filter
# ==============================================================================
def test_preprocess_for_parsing_filter():
    """
    Test that the `utils.preprocessing.preprocess` works for parsing filter.
    """

    # Check parsing filter.
    dict_of_preprocessed_texts = preprocess(
        dict_of_texts={
            "0": "Comment signaler une perte de carte de paiement ?",
            "1": "Quelle est la procédure pour chercher une carte de crédit avalée ?",
            "2": "Ma carte Visa a un plafond de paiment trop bas, puis-je l'augmenter ?",
        },
        apply_parsing_filter=True,
    )

    # Assertions
    assert dict_of_preprocessed_texts
    assert sorted(dict_of_preprocessed_texts.keys()) == ["0", "1", "2"]
    assert dict_of_preprocessed_texts["0"] == "comment signaler perte"
    assert dict_of_preprocessed_texts["1"] == "quelle est la procedure chercher"
    assert dict_of_preprocessed_texts["2"] == "carte a plafond l"


# ==============================================================================
# test_preprocess_for_lemmatization
# ==============================================================================
def test_preprocess_for_lemmatization():
    """
    Test that the `utils.preprocessing.preprocess` works for lemmatization.
    """

    # Check lemmatization.
    dict_of_preprocessed_texts = preprocess(
        dict_of_texts={
            "0": "Comment signaler une perte de carte de paiement ?",
            "1": "Quelle est la procédure pour chercher une carte de crédit avalée ?",
            "2": "Ma carte Visa a un plafond de paiment trop bas, puis-je l'augmenter ?",
        },
        apply_lemmatization=True,
    )

    # Assertions
    assert dict_of_preprocessed_texts
    assert sorted(dict_of_preprocessed_texts.keys()) == ["0", "1", "2"]
    assert dict_of_preprocessed_texts["0"] == "comment signaler un perte de carte de paiement"
    assert dict_of_preprocessed_texts["1"] == "quell etre le procedure pour chercher un carte de credit avaler"
    assert dict_of_preprocessed_texts["2"] == "mon carte visa avoir un plafond de paiment trop bas puis je l augmenter"
