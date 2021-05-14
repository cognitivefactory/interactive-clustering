# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.utils.preprocessing
* Description:  Utilities methods to apply NLP preprocessing.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

# Import path management dependencies.

# Import preprocessing dependencies.
import unicodedata

# Python code typing (mypy).
from typing import Dict

# import spacy
import fr_core_news_sm

# from nltk.stem.snowball import SnowballStemmer


# ==============================================================================
# NLP PREPROCESSING
# ==============================================================================
def preprocess(
    dict_of_texts: Dict[str, str],
    apply_stopwords_deletion: bool = False,
    apply_parsing_filter: bool = False,
    apply_lemmatization: bool = False,
    # TODO language (str, optional) : #TODO. Defaults to `"fr"`.
) -> Dict[str, str]:
    """
    A method used to preprocess texts.
    It applies simple preprocessing (lowercasing, punctuations deletion, accents replacement, whitespace deletion).
    Some options are available to delete stopwords, apply lemmatization, and delete tokens according to their depth in the denpendency tree.

    References:
        - _spaCy_: `Honnibal, M. et I. Montani (2017). spaCy 2 : Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.`
        - _spaCy_ language model `fr_core_news_sm`: `https://spacy.io/usage/models`
        - _NLTK_: `Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.`
        - _NLTK_ _'SnowballStemmer'_: `https://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball`

    Args:
        dict_of_texts (Dict[str,str]): A dictionary that contains the texts to preprocess.
        apply_stopwords_deletion (bool, optional): The option to delete stopwords. Defaults to `False`.
        apply_parsing_filter (bool, optional): The option to filter tokens based on dependency parsing results. If set, it only keeps `"ROOT"` tokens and their direct children. Defaults to `False`.
        apply_lemmatization (bool, optional): The option to lemmatize tokens. Defaults to `False`.

    Returns:
        Dict[str,str]: A dictionary that contains the preprocessed texts.
    """

    # Initialize dictionary of preprocessed texts.
    dict_of_preprocessed_texts: Dict[str, str] = {}

    # Initialize punctuation translator.
    punctuation_translator = str.maketrans(
        {
            punct: " "
            for punct in (
                ".",
                ",",
                ";",
                ":",
                "!",
                "¡",
                "?",
                "¿",
                "…",
                "•",
                "(",
                ")",
                "{",
                "}",
                "[",
                "]",
                "«",
                "»",
                "^",
                "`",
                "'",
                '"',
                "\\",
                "/",
                "|",
                "-",
                "_",
                "#",
                "&",
                "~",
                "@",
            )
        }
    )

    # Load vectorizer (spacy language model).
    # TODO: Use fr_core_news_sm or spacy
    # path_to_model: str = os.path.dirname(os.path.realpath(__file__)) + "/" + "fr_core_news_sm-2.3.0/"
    # spacy_nlp = spacy.load(
    #     name=path_to_model,
    #     disable=[
    #         # "tagger", # Needed for lemmatization.
    #         # "parser", # Needed for filtering on dependency parsing.
    #         "ner",  # Not needed
    #     ],
    # )
    spacy_nlp = fr_core_news_sm.load()

    # Initialize stemmer.
    ####stemmer = SnowballStemmer(language="french")

    # For each text...
    for key, text in dict_of_texts.items():

        # Apply lowercasing.
        preprocessed_text: str = text.lower()

        # Apply punctuation deletion (before tokenization).
        preprocessed_text = preprocessed_text.translate(punctuation_translator)

        # Apply tokenization and spaCy pipeline.
        tokens = [
            token
            for token in spacy_nlp(preprocessed_text)
            if (
                # Spaces are not allowed.
                not token.is_space
            )
            and (
                # Punctuation are not allowed.
                not token.is_punct
                and not token.is_quote
            )
            and (
                # If set, stopwords are not allowed.
                (not apply_stopwords_deletion)
                or (not token.is_stop)
            )
            and (
                # If set, stopwords are not allowed.
                (not apply_parsing_filter)
                or (len(list(token.ancestors)) <= 1)
            )
        ]

        # Apply retokenization with lemmatization.
        if apply_lemmatization:

            preprocessed_text = " ".join([token.lemma_.strip() for token in tokens])

        # Apply retokenization without lemmatization.
        else:
            preprocessed_text = " ".join([token.text.strip() for token in tokens])

        # Apply accents deletion (after lemmatization).
        preprocessed_text = "".join(
            [char for char in unicodedata.normalize("NFKD", preprocessed_text) if not unicodedata.combining(char)]
        )

        # Store preprocessed text.
        dict_of_preprocessed_texts[key] = preprocessed_text

    return dict_of_preprocessed_texts
