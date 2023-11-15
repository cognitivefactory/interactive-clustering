# Usage

Import dependencies.
```python
from cognitivefactory.interactive_clustering.utils.preprocessing import preprocess
from cognitivefactory.interactive_clustering.utils.vectorization import vectorize
from cognitivefactory.interactive_clustering.constraints.factory import managing_factory
from cognitivefactory.interactive_clustering.clustering.factory import clustering_factory
from cognitivefactory.interactive_clustering.sampling.factory import sampling_factory
```

## Initialization step (iteration `0`)

Get data.
```python
# Define dictionary of texts.
dict_of_texts = {
    "0": "This is my first question.",
    "1": "This is my second item.",
    "2": "This is my third query.",
    "3": "This is my fourth issue.",
    # ...
    "N": "This is my last request.",
}
```

Preprocess data.
```python
# Preprocess data.
dict_of_preprocess_texts = preprocess(
    dict_of_texts=dict_of_texts,
    spacy_language_model="fr_core_news_md",
)  # Apply simple preprocessing. Spacy language model has to be installed. Other parameters are available.
```

Vectorize data.
```python
# Vectorize data.
dict_of_vectors = vectorize(
    dict_of_texts=dict_of_preprocess_texts,
    vectorizer_type="tfidf",
)  # Apply TF-IDF vectorization. Other parameters are available.
```

Initialize constraints manager.
```python
# Create an instance of binary constraints manager.
constraints_manager = managing_factory(
    manager="binary",
    list_of_data_IDs = list(dict_of_texts.keys()),
)
```

Apply first clustering without constraints.
```python
# Create an instance of constrained COP-kmeans clustering.
clustering_model = clustering_factory(
    algorithm="kmeans",
    random_seed=1,
)  # Other clustering algorithms are available.

# Run clustering.
clustering_result = clustering_model.cluster(
    constraints_manager=constraints_manager,
    nb_clusters=2,
    vectors=dict_of_vectors,
)
```

## Iteration step (iteration `N`)

Check if all possible constraints are annotated.
```python
# Check if all constraints are already annotated.
is_finish = constraints_manager.check_completude_of_constraints()

# Print result
if is_finish:
    print("All possible constraints are annotated. No more iteration can be run.")
    # break
```

Sampling constraints to annotate.
```python
# Create an instance of random sampler.
sampler = sampling_factory(
    algorithm="random",
    random_seed=None,
)  # Other algorithms are available.

# Sample constraints to annotated.
selection = sampler.sample(
    constraints_manager=constraints_manager,
    nb_to_select=3,
    #clustering_result=clustering_result,  # Results from iteration `N-1`.
    #vectors=dict_of_vectors,
)
```

Annotate constraints (manual operation).
```python
# TODO: Use a graphical interface for interactive clustering.
# WIP: Project `interactive-clustering-gui`.

list_of_annotation = []  # List of triplets with format `(data_ID1, data_ID2, annotation_type)` where `annotation_type` can be "MUST_LINK" or "CANNOT_LINK".
```

Update constraints manager.
```python
for annotation in list_of_annotation:

    # Get the annotation
    data_ID1, data_ID2, constraint_type = annotation

    # Add constraints
    try:
        constraints_manager.add_constraint(
            data_ID1=data_ID1,
            data_ID2=data_ID2,
            constraint_type=constraint_type
        )
    except ValueError as err:
        print(err)  # An error can occur if parameters are incorrect or if annotation is incompatible with previous annotation.
```

Determine the range of possible cluster number.
```python
# Get min and max range of clusters based on constraints.
min_n, max_n = constraints_manager.get_min_and_max_number_of_clusters()

# Choose the number of cluster.
nb_clusters = int( (min_n + max_n) / 2 ) # or manual selection.
```

Run constrained clustering.
```python
# Create an instance of constrained COP-kmeans clustering.
clustering_model = clustering_factory(
    algorithm="kmeans",
    random_seed=1,
)  # Other clustering algorithms are available.

# Run clustering.
clustering_result = clustering_model.cluster(
    constraints_manager=constraints_manager,  # Annotation since iteration `0`.
    nb_clusters=nb_clusters,
    vectors=dict_of_vectors,
)  # Clustering results are corrected since the previous iteration.
```

Analyze cluster (not implemented here).
```python
# TODO: Evaluate completness, homogeneity, v-measure, rand index (basic, adjusted), mutual information (basic, normalized, mutual), ...
# TODO: Plot clustering.
```
