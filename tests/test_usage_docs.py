# -*- coding: utf-8 -*-

"""
* Name:         interactive-clustering/tests/tests_docs.py
* Description:  Unittests for the documentation.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""


# ==============================================================================
# test_docs_usage :
# ==============================================================================
def test_docs_usage():
    """
    Test the `usage` documentation.
    """

    # Import dependencies.
    from cognitivefactory.interactive_clustering.clustering.factory import (  # noqa: C0415 (not top level import, it's fine)
        clustering_factory,
    )
    from cognitivefactory.interactive_clustering.constraints.factory import (  # noqa: C0415 (not top level import, it's fine)
        managing_factory,
    )
    from cognitivefactory.interactive_clustering.sampling.factory import (  # noqa: C0415 (not top level import, it's fine)
        sampling_factory,
    )
    from cognitivefactory.interactive_clustering.utils.preprocessing import (  # noqa: C0415 (not top level import, it's fine)
        preprocess,
    )
    from cognitivefactory.interactive_clustering.utils.vectorization import (  # noqa: C0415 (not top level import, it's fine)
        vectorize,
    )

    ###
    ### Initialization step (iteration `0`)
    ###
    # Define dictionary of texts.
    dict_of_texts = {
        "0": "This is my first question.",
        "1": "This is my second item.",
        "2": "This is my third query.",
        "3": "This is my fourth issue.",
        # ...
        "N": "This is my last request.",
    }

    # Preprocess data.
    dict_of_preprocess_texts = preprocess(
        dict_of_texts=dict_of_texts,
        spacy_language_model="fr_core_news_sm",
    )  # Apply simple preprocessing. Spacy language model has to be installed. Other parameters are available.

    # Vectorize data.
    dict_of_vectors = vectorize(
        dict_of_texts=dict_of_preprocess_texts,
        vectorizer_type="tfidf",
    )  # Apply TF-IDF vectorization. Other parameters are available.

    # Create an instance of binary constraints manager.
    constraints_manager = managing_factory(
        manager="binary",
        list_of_data_IDs=list(dict_of_texts.keys()),
    )
    assert constraints_manager

    # Create an instance of constrained COP-kmeans clustering.
    clustering_model = clustering_factory(
        algorithm="kmeans",
        random_seed=1,
    )  # Other clustering algorithms are available.
    assert clustering_model

    # Run clustering.
    clustering_result = clustering_model.cluster(
        constraints_manager=constraints_manager,
        vectors=dict_of_vectors,
        nb_clusters=2,
    )
    assert clustering_result

    ###
    ### Iteration step (iteration `N`)
    ###

    # Check if all constraints are already annotated.
    is_finish = constraints_manager.check_completude_of_constraints()

    # Print result
    if is_finish:  # pragma: no cover
        print("All possible constraints are annotated. No more iteration can be run.")
        # break

    # Create an instance of random sampler.
    sampler = sampling_factory(
        algorithm="random",
        random_seed=None,
    )  # Other algorithms are available.

    # Sample constraints to annotated.
    selection = sampler.sample(
        constraints_manager=constraints_manager,
        nb_to_select=3,
        # clustering_result=clustering_result,  # Results from iteration `N-1`.
        # vectors=dict_of_vectors,
    )
    assert len(selection) == 3
    assert selection

    # Annotate constraints (manual operation).
    ANNOTATIONS = ["MUST_LINK", "CANNOT_LINK", None]
    list_of_annotation = [
        (data_ID1, data_ID2, ANNOTATIONS[i]) for i, (data_ID1, data_ID2) in enumerate(selection)
    ]  # List of triplets with format `(data_ID1, data_ID2, annotation_type)` where `annotation_type` can be "MUST_LINK" or "CANNOT_LINK".

    for annotation in list_of_annotation:

        # Get the annotation
        data_ID1, data_ID2, constraint_type = annotation

        # Add constraints
        try:
            constraints_manager.add_constraint(data_ID1=data_ID1, data_ID2=data_ID2, constraint_type=constraint_type)
        except ValueError as err:
            print(
                err
            )  # An error can occur if parameters are incorrect or if annotation is incompatible with previous annotation.

    # Get min and max range of clusters based on constraints.
    min_n, max_n = constraints_manager.get_min_and_max_number_of_clusters()

    # Choose the number of cluster.
    nb_clusters = int((min_n + max_n) / 2)  # or manual selection.

    # Create an instance of constrained COP-kmeans clustering.
    clustering_model = clustering_factory(
        algorithm="kmeans",
        random_seed=1,
    )  # Other clustering algorithms are available.
    assert clustering_model

    # Run clustering.
    clustering_result = clustering_model.cluster(
        constraints_manager=constraints_manager,  # Annotation since iteration `0`.
        nb_clusters=nb_clusters,
        vectors=dict_of_vectors,
    )  # Clustering results are corrected since the previous iteration.
    assert clustering_result
