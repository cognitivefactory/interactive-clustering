import json
import random
import time
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from sklearn import metrics

from cognitivefactory.interactive_clustering.clustering.abstract import rename_clusters_by_order
from cognitivefactory.interactive_clustering.clustering.kmeans import KMeansConstrainedClustering
from cognitivefactory.interactive_clustering.constraints.binary import BinaryConstraintsManager
from cognitivefactory.interactive_clustering.utils.vectorization import vectorize
from src.cognitivefactory.interactive_clustering.clustering.affinity_propagation import (
    AffinityPropagationConstrainedClustering,
)
from src.cognitivefactory.interactive_clustering.clustering.dbscan import DBScanConstrainedClustering
from src.cognitivefactory.interactive_clustering.clustering.mpckmeans import MPCKMeansConstrainedClustering


def save_results(results, dst_path):
    """
    Saves in a .json file results contained in a variable.

    Args:
        results : Variable containing the results to save.
        dst_path : Path of the .json file were the results will be saved.
    """
    with open(dst_path, "w") as json_file:
        json.dump(results, json_file)
        json_file.close()


def load_results(src_path):
    """
    Loads results saved in a .json file.

    Args:
        src_path : Path of the .json file.

    Returns:
        Results loaded.
    """

    with open(src_path, "r") as json_file:
        results = json.load(json_file)
        json_file.close()

    return results


def load_dataset(
    dataset_path="./French_trainset_for_chatbots_dealing_with_usual_requests_on_bank_cards.csv",
    dict_of_preprocessed_texts_path="./dict_of_preprocessed_texts.json",
    desired_size=None,
    list_of_labels=[
        "alerte_perte_vol_carte",
        "carte_avalee",
        "commande_carte",
        "consultation_solde",
        "couverture_assurrance",
        "deblocage_carte",
        "gestion_carte_virtuelle",
        "gestion_decouvert",
        "gestion_plafond",
        "gestion_sans_contact",
    ],
):
    """
    Loads a labelled dataset of sentences and returns a dict containing the corresponding vectors,
     and a dict indicating the labels.

    Args:
        dataset_path : Path of the dataset to load.
        dict_of_preprocessed_texts_path : Path of the preprocessed texts saved in a .json file.
        desired_size : Desired size for the dataset (if None, all the dataset).
        list_of_labels : List of all the labels of the dataset.

    Returns:
        Tuple(Dict[str,csr_matrix], Dict[str,int]) : A couple of dictionary that contain the vectorized sentences
         for the first one, and the real cluster (as integer) for each sentence for the second one.
    """
    ###
    ### Load dataset
    ###

    df = pd.read_csv(dataset_path)
    dataset_array = df.to_numpy()

    ###
    ### Store questions and labels in dict
    ###

    dict_of_questions: Dict[str, str] = {}
    dict_of_labels: Dict[str, str] = {}

    size = dataset_array.shape[0]
    if desired_size:
        size = min(dataset_array.shape[0], desired_size)
    size = max(size, len(list_of_labels))

    dict_of_classes_points_indexes = {label: [] for label in list_of_labels}

    for i in range(dataset_array.shape[0]):
        dict_of_classes_points_indexes[dataset_array[i, 1]].append(i)

    list_of_kept_indexes = []
    classes_size = int(size / len(list_of_labels))
    last_class_size = classes_size + size % len(list_of_labels)

    for label in list_of_labels[:-1]:
        for k in range(classes_size):
            list_of_kept_indexes.append(dict_of_classes_points_indexes[label][k])

    for k in range(last_class_size):
        list_of_kept_indexes.append(dict_of_classes_points_indexes[list_of_labels[-1]][k])

    for index in list_of_kept_indexes:
        dict_of_questions[str(index)] = dataset_array[index, 0]
        dict_of_labels[str(index)] = dataset_array[index, 1]

    ###
    ### Apply preprocessing and vectorization
    ###

    dict_of_preprocessed_texts = load_results(dict_of_preprocessed_texts_path)
    dict_of_kept_preprocessed_texts = {
        question: dict_of_preprocessed_texts[question] for question in dict_of_questions.values()
    }

    dict_of_vectors = vectorize(dict_of_texts=dict_of_kept_preprocessed_texts)

    ###
    ### Rewrite labels as int to match with the output of the clustering algorithms
    ###

    dict_of_int_labels: Dict[str, int] = {
        list(dict_of_labels.values())[k]: k for k in range(len(dict_of_labels.values()))
    }
    dict_of_int_labels = rename_clusters_by_order(dict_of_int_labels)

    ###
    ### Get the dict of the real clusters (to compare with predicted)
    ###

    dict_of_real_clusters: Dict[str, int] = {
        dict_of_questions[key]: dict_of_int_labels[dict_of_labels[key]] for key in dict_of_labels.keys()
    }
    dict_of_real_clusters = rename_clusters_by_order(dict_of_real_clusters)

    return dict_of_vectors, dict_of_real_clusters


def estimate_mean_min_distance_between_same_cluster_points(dict_of_vectors, dict_of_real_clusters):
    """
    Estimates the mean minimum distance between points of a same cluster.
    (This estimation can be useful for algorithm that needs as hyperparameter a radius
     to define the concept of neighborhood (e.g. C-DBScan))

    Args:
        dict_of_vectors :  Dictionary containing the vector describing each data point.
        dict_of_real_clusters : Dictionary containing the real clustering of each point.

    Returns:
        Float : Mean of the minimum distances between two different points of the same cluster.
    """

    dict_of_same_cluster_points: Dict[str, List[str]] = {value: [] for value in dict_of_real_clusters.values()}

    for point in dict_of_real_clusters.keys():
        dict_of_same_cluster_points[dict_of_real_clusters[point]].append(point)

    dict_of_min_distances: Dict[str, float] = {}

    for point in dict_of_same_cluster_points.keys():
        list_of_points = dict_of_same_cluster_points[point]
        list_of_distances = []

        for i in range(len(list_of_points)):
            for j in range(len(list_of_points)):
                if i != j:
                    list_of_distances.append(
                        distance.euclidean(
                            dict_of_vectors[list_of_points[i]].toarray()[0],
                            dict_of_vectors[list_of_points[j]].toarray()[0],
                        )
                    )

        if list_of_distances:
            dict_of_min_distances[point] = min(list_of_distances)

    mean_min_distance = mean(list(dict_of_min_distances.values()))

    return mean_min_distance


def get_constraints_couples(dict_of_real_clusters):
    """
    Computes all the possible Must-link and cannot-link constraints regarding a clustering of points.

    Args:
        dict_of_real_clusters : Dictionary containing the real cluster for each point.

    Returns:
        Dict[Tuple(str, str),str] : A dictionary that contains as keys each couple involved in a possible constraint,
         and as values the types of the constraints.
    """

    ###
    ### Get all the possible Must-link and Cannot-link constraints regarding the dataset
    ###

    dict_of_constraints_couples: Dict[Tuple[str, str], str] = {}

    list_of_points = list(dict_of_real_clusters.keys())

    for i in range(len(list_of_points)):
        for j in range(i + 1, len(list_of_points)):
            if dict_of_real_clusters[list_of_points[i]] == dict_of_real_clusters[list_of_points[j]]:
                dict_of_constraints_couples[(list_of_points[i], list_of_points[j])] = "MUST_LINK"
            else:
                dict_of_constraints_couples[(list_of_points[i], list_of_points[j])] = "CANNOT_LINK"

    return dict_of_constraints_couples


def run_performances_measure(
    clustering_model,
    dict_of_vectors,
    dict_of_real_clusters,
    dict_of_constraints_couples,
    number_of_iterations=250,
    number_of_clusters=10,
    specific_nb_of_clusters=False
):
    """
    Runs iteratively clustering by adding randomly some constraints at each iteration.
    Homogeneity, completeness, V-measure, execution time and number of clusters are measured for each clustering.

    Args:
        clustering_model : Instance of the clustering model.
        dict_of_vectors :  Dictionary containing the vector describing each data point.
        dict_of_real_clusters : Dictionary containing the real clustering of each point.
        dict_of_constraints_couples : Dictionary containing all the couples of points involved in a possible constraint.
        number_of_iterations : Number of iteration of the measure.
        number_of_clusters : Number of clusters given as hyperparameter to the clustering model.
        specific_nb_of_clusters : Boolean to indicate if at each iteration, an attribute 'number_of_clusters' of
        the clustering model precise the number of clusters.

    Returns:
        Dict[int,Dict[str, float] : A dictionary that contains as keys the iteration numbers,
         and as values  dictionaries containing the associated measured homogeneity, completeness, V-measuree,
         execution time, number of clusters, and dict of predicted cluster.
    """
    start_time = time.time()
    number_of_constraints = len(dict_of_constraints_couples)
    constraints_increment = int(number_of_constraints / number_of_iterations)
    nb_clusters = number_of_clusters

    # Dict to store the results
    dict_of_clustering_performances: Dict[str, Dict[str, float]] = {
        str(iteration): {} for iteration in range(number_of_iterations + 1)
    }

    # Create a constraints manager instance
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(dict_of_vectors.keys()))

    # Apply clustering

    start_time_clustering = time.time()
    dict_of_predicted_clusters = clustering_model.cluster(
        constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
    )
    clustering_time = time.time() - start_time_clustering

    # Compute homogeneity.
    dict_of_clustering_performances[str(0)]["homogeneity"] = metrics.homogeneity_score(
        list(dict_of_real_clusters.values()), list(dict_of_predicted_clusters.values())
    )

    # Compute completeness.
    dict_of_clustering_performances[str(0)]["completeness"] = metrics.completeness_score(
        list(dict_of_real_clusters.values()), list(dict_of_predicted_clusters.values())
    )

    # Compute v_measure.
    dict_of_clustering_performances[str(0)]["v_measure"] = metrics.v_measure_score(
        list(dict_of_real_clusters.values()), list(dict_of_predicted_clusters.values())
    )

    # Store clustering time.
    dict_of_clustering_performances[str(0)]["time"] = clustering_time

    # Store number of clusters.
    if specific_nb_of_clusters:
        nb_clusters = clustering_model.number_of_clusters

    dict_of_clustering_performances[str(0)]["nb_clusters"] = nb_clusters

    # Store predicted clusters.
    dict_of_clustering_performances[str(0)]["predicted_clusters"] = dict_of_predicted_clusters

    for iteration in range(1, number_of_iterations + 1):

        # Shuffle the remaining constraints
        l = list(dict_of_constraints_couples.items())
        random.shuffle(l)

        # Add constraints to the constraints manager

        dict_of_applied_constraints = dict([l[k] for k in range(constraints_increment)])

        for couple in dict_of_applied_constraints.keys():
            constraints_manager.add_constraint(
                data_ID1=couple[0], data_ID2=couple[1], constraint_type=dict_of_applied_constraints[couple]
            )

        # Remove added constraints from dict storing remaining unadded constraints
        dict_of_constraints_couples = dict(l[constraints_increment:])

        # Apply clustering

        start_time_clustering = time.time()
        dict_of_predicted_clusters = clustering_model.cluster(
            constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
        )
        clustering_time = time.time() - start_time_clustering

        # Compute homogeneity.
        dict_of_clustering_performances[str(iteration)]["homogeneity"] = metrics.homogeneity_score(
            list(dict_of_real_clusters.values()), list(dict_of_predicted_clusters.values())
        )

        # Compute completeness.
        dict_of_clustering_performances[str(iteration)]["completeness"] = metrics.completeness_score(
            list(dict_of_real_clusters.values()), list(dict_of_predicted_clusters.values())
        )

        # Compute v_measure.
        dict_of_clustering_performances[str(iteration)]["v_measure"] = metrics.v_measure_score(
            list(dict_of_real_clusters.values()), list(dict_of_predicted_clusters.values())
        )

        # Store clustering time.
        dict_of_clustering_performances[str(iteration)]["time"] = clustering_time

        # Store number of clusters.
        if specific_nb_of_clusters:
            nb_clusters = clustering_model.number_of_clusters

        dict_of_clustering_performances[str(iteration)]["nb_clusters"] = nb_clusters

        # Store predicted clusters.
        dict_of_clustering_performances[str(iteration)]["predicted_clusters"] = dict_of_predicted_clusters

        print("Iteration " + str(iteration) + "/" + str(number_of_iterations))

        current_time = time.time() - start_time

        hours = int(current_time / 3600)
        minuts = int((int(current_time) % 3600) / 60)
        seconds = int(current_time) % 60

        print("--- " + str(hours) + "h " + str(minuts) + "min " + str(seconds) + "s ---")

    return dict_of_clustering_performances


def run_all_performances_measures(
    dict_of_vectors,
    dict_of_real_clusters,
    dict_of_constraints_couples,
    number_of_iterations=250,
    number_of_clusters=10,
    kmeans=True,
    c_dbscan=True,
    mpckmeans=True,
    affinity_propagation=True,
):
    """
    Runs iteratively clustering for all available models by adding randomly some constraints at each iteration.
    Homogeneity, completeness, V-measure, execution time, number of clusters are measured for each clustering.

    Args:
        dict_of_vectors :  Dictionary containing the vector describing each data point.
        dict_of_real_clusters : Dictionary containing the real clustering of each point.
        dict_of_constraints_couples : Dictionary containing all the couples of points involved in a possible constraint.
        number_of_iterations : Number of iteration of the measure.
        number_of_clusters : Number of clusters given as hyperparameter to the clustering model.
        kmeans : Boolean to consider or not Kmeans.
        c_dbscan : Boolean to consider or not C-DBScan.
        mpckmeans : Boolean to consider or not MPCKmeans.
        affinity_propagation : Boolean to consider or not Affinity propagation.

    Returns:
        Dict[str, Dict[str, Dict]] : A dictionary that contains for each model a dictionary with as keys the
         iteration numbers, and as values dictionaries containing the associated measured homogeneity,
          completeness, V-measuree, execution time, number of clusters, and dict of predicted cluster.
    """
    start_time = time.time()
    number_of_constraints = len(dict_of_constraints_couples)
    constraints_increment = int(number_of_constraints / number_of_iterations)

    # Dict to store all the results
    dict_of_clustering_performances = {"kmeans": {}, "c_dbscan": {}, "mpckmeans": {}, "affinity_propagation": {}}

    for key in dict_of_clustering_performances.keys():
        dict_of_clustering_performances[key] = {str(iteration): {} for iteration in range(number_of_iterations + 1)}

    # Create a constraints manager instance
    constraints_manager = BinaryConstraintsManager(list_of_data_IDs=list(dict_of_vectors.keys()))

    # Initialize the clustering models

    kmeans_model = KMeansConstrainedClustering()

    suitable_eps = estimate_mean_min_distance_between_same_cluster_points(dict_of_vectors, dict_of_real_clusters)
    c_dbscan_model = DBScanConstrainedClustering(eps=suitable_eps, min_samples=8)

    mpckmeans_model = MPCKMeansConstrainedClustering(max_iteration=15)

    affinity_propagation_model = AffinityPropagationConstrainedClustering(
        absolute_must_links=True,
        smart_preference=False,
        smart_preference_factor=2 / 3,
        ensure_nb_clusters=True,
        max_iteration=500,
    )

    if kmeans:

        # Apply clustering with Kmeans

        start_time_kmeans = time.time()
        kmeans_dict_of_predicted_clusters = kmeans_model.cluster(
            constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
        )
        kmeans_time = time.time() - start_time_kmeans

        # Compute homogeneity, completeness and V-measure.

        dict_of_clustering_performances["kmeans"][str(0)]["homogeneity"] = metrics.homogeneity_score(
            list(dict_of_real_clusters.values()), list(kmeans_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["kmeans"][str(0)]["completeness"] = metrics.completeness_score(
            list(dict_of_real_clusters.values()), list(kmeans_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["kmeans"][str(0)]["v_measure"] = metrics.v_measure_score(
            list(dict_of_real_clusters.values()), list(kmeans_dict_of_predicted_clusters.values())
        )

        # Store clustering time.

        dict_of_clustering_performances["kmeans"][str(0)]["time"] = kmeans_time

        # Store number of clusters.

        dict_of_clustering_performances["kmeans"][str(0)]["nb_clusters"] = number_of_clusters

        # Store predicted clusters.

        dict_of_clustering_performances["kmeans"][str(0)]["predicted_clusters"] = kmeans_dict_of_predicted_clusters

    if c_dbscan:

        # Apply clustering with C-DBScan

        start_time_c_dbscan = time.time()
        c_dbscan_dict_of_predicted_clusters = c_dbscan_model.cluster(
            constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
        )
        c_dbscan_time = time.time() - start_time_c_dbscan

        # Compute homogeneity, completeness and V-measure.

        dict_of_clustering_performances["c_dbscan"][str(0)]["homogeneity"] = metrics.homogeneity_score(
            list(dict_of_real_clusters.values()), list(c_dbscan_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["c_dbscan"][str(0)]["completeness"] = metrics.completeness_score(
            list(dict_of_real_clusters.values()), list(c_dbscan_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["c_dbscan"][str(0)]["v_measure"] = metrics.v_measure_score(
            list(dict_of_real_clusters.values()), list(c_dbscan_dict_of_predicted_clusters.values())
        )

        # Store clustering time.

        dict_of_clustering_performances["c_dbscan"][str(0)]["time"] = c_dbscan_time

        # Store number of clusters.

        dict_of_clustering_performances["c_dbscan"][str(0)]["nb_clusters"] = c_dbscan_model.number_of_clusters

        # Store predicted clusters.

        dict_of_clustering_performances["c_dbscan"][str(0)]["predicted_clusters"] = c_dbscan_dict_of_predicted_clusters

    if mpckmeans:

        # Apply clustering with MPCKmeans

        start_time_mpckmeans = time.time()
        mpckmeans_dict_of_predicted_clusters = mpckmeans_model.cluster(
            constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
        )
        mpckmeans_time = time.time() - start_time_mpckmeans

        # Compute homogeneity, completeness and V-measure.

        dict_of_clustering_performances["mpckmeans"][str(0)]["homogeneity"] = metrics.homogeneity_score(
            list(dict_of_real_clusters.values()), list(mpckmeans_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["mpckmeans"][str(0)]["completeness"] = metrics.completeness_score(
            list(dict_of_real_clusters.values()), list(mpckmeans_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["mpckmeans"][str(0)]["v_measure"] = metrics.v_measure_score(
            list(dict_of_real_clusters.values()), list(mpckmeans_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["mpckmeans"][str(0)]["time"] = mpckmeans_time

        # Store number of clusters.

        dict_of_clustering_performances["mpckmeans"][str(0)]["nb_clusters"] = number_of_clusters

        # Store predicted clusters.

        dict_of_clustering_performances["mpckmeans"][str(0)][
            "predicted_clusters"
        ] = mpckmeans_dict_of_predicted_clusters

    if affinity_propagation:

        # Apply clustering with Affinity propagation

        start_time_affinity_propagation = time.time()
        affinity_propagation_dict_of_predicted_clusters = affinity_propagation_model.cluster(
            constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
        )
        affinity_propagation_time = time.time() - start_time_affinity_propagation

        # Compute homogeneity, completeness and V-measure.

        dict_of_clustering_performances["affinity_propagation"][str(0)]["homogeneity"] = metrics.homogeneity_score(
            list(dict_of_real_clusters.values()), list(affinity_propagation_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["affinity_propagation"][str(0)]["completeness"] = metrics.completeness_score(
            list(dict_of_real_clusters.values()), list(affinity_propagation_dict_of_predicted_clusters.values())
        )

        dict_of_clustering_performances["affinity_propagation"][str(0)]["v_measure"] = metrics.v_measure_score(
            list(dict_of_real_clusters.values()), list(affinity_propagation_dict_of_predicted_clusters.values())
        )

        # Store clustering time.

        dict_of_clustering_performances["affinity_propagation"][str(0)]["time"] = affinity_propagation_time

        # Store number of clusters.

        dict_of_clustering_performances["affinity_propagation"][str(0)][
            "nb_clusters"
        ] = affinity_propagation_model.number_of_clusters

        # Store predicted clusters.

        dict_of_clustering_performances["affinity_propagation"][str(0)][
            "predicted_clusters"
        ] = affinity_propagation_dict_of_predicted_clusters

    for iteration in range(1, number_of_iterations + 1):

        # Shuffle the remaining constraints
        l = list(dict_of_constraints_couples.items())
        random.shuffle(l)

        # Add constraints to the constraints manager

        dict_of_applied_constraints = dict([l[k] for k in range(constraints_increment)])

        for couple in dict_of_applied_constraints.keys():
            constraints_manager.add_constraint(
                data_ID1=couple[0], data_ID2=couple[1], constraint_type=dict_of_applied_constraints[couple]
            )

        # Remove added constraints from dict storing remaining unadded constraints
        dict_of_constraints_couples = dict(l[constraints_increment:])

        if kmeans:
            # Apply clustering with Kmeans

            start_time_kmeans = time.time()
            kmeans_dict_of_predicted_clusters = kmeans_model.cluster(
                constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
            )
            kmeans_time = time.time() - start_time_kmeans

            # Compute homogeneity, completeness and V-measure.

            dict_of_clustering_performances["kmeans"][str(iteration)]["homogeneity"] = metrics.homogeneity_score(
                list(dict_of_real_clusters.values()), list(kmeans_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["kmeans"][str(iteration)]["completeness"] = metrics.completeness_score(
                list(dict_of_real_clusters.values()), list(kmeans_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["kmeans"][str(iteration)]["v_measure"] = metrics.v_measure_score(
                list(dict_of_real_clusters.values()), list(kmeans_dict_of_predicted_clusters.values())
            )

            # Store clustering time.

            dict_of_clustering_performances["kmeans"][str(iteration)]["time"] = kmeans_time

            # Store number of clusters.

            dict_of_clustering_performances["kmeans"][str(iteration)]["nb_clusters"] = number_of_clusters

            # Store predicted clusters.

            dict_of_clustering_performances["kmeans"][str(iteration)][
                "predicted_clusters"
            ] = kmeans_dict_of_predicted_clusters

        if c_dbscan:
            # Apply clustering with C-DBScan

            start_time_c_dbscan = time.time()
            c_dbscan_dict_of_predicted_clusters = c_dbscan_model.cluster(
                constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
            )
            c_dbscan_time = time.time() - start_time_c_dbscan

            # Compute homogeneity, completeness and V-measure.

            dict_of_clustering_performances["c_dbscan"][str(iteration)]["homogeneity"] = metrics.homogeneity_score(
                list(dict_of_real_clusters.values()), list(c_dbscan_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["c_dbscan"][str(iteration)]["completeness"] = metrics.completeness_score(
                list(dict_of_real_clusters.values()), list(c_dbscan_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["c_dbscan"][str(iteration)]["v_measure"] = metrics.v_measure_score(
                list(dict_of_real_clusters.values()), list(c_dbscan_dict_of_predicted_clusters.values())
            )

            # Store clustering time.

            dict_of_clustering_performances["c_dbscan"][str(iteration)]["time"] = c_dbscan_time

            # Store number of clusters.

            dict_of_clustering_performances["c_dbscan"][str(iteration)][
                "nb_clusters"
            ] = c_dbscan_model.number_of_clusters

            # Store predicted clusters.

            dict_of_clustering_performances["c_dbscan"][str(iteration)][
                "predicted_clusters"
            ] = c_dbscan_dict_of_predicted_clusters

        if mpckmeans:
            # Apply clustering with MPCKmeans

            start_time_mpckmeans = time.time()
            mpckmeans_dict_of_predicted_clusters = mpckmeans_model.cluster(
                constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
            )
            mpckmeans_time = time.time() - start_time_mpckmeans

            # Compute homogeneity, completeness and V-measure.

            dict_of_clustering_performances["mpckmeans"][str(iteration)]["homogeneity"] = metrics.homogeneity_score(
                list(dict_of_real_clusters.values()), list(mpckmeans_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["mpckmeans"][str(iteration)]["completeness"] = metrics.completeness_score(
                list(dict_of_real_clusters.values()), list(mpckmeans_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["mpckmeans"][str(iteration)]["v_measure"] = metrics.v_measure_score(
                list(dict_of_real_clusters.values()), list(mpckmeans_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["mpckmeans"][str(iteration)]["time"] = mpckmeans_time

            # Store number of clusters.

            dict_of_clustering_performances["mpckmeans"][str(iteration)]["nb_clusters"] = number_of_clusters

            # Store predicted clusters.

            dict_of_clustering_performances["mpckmeans"][str(iteration)][
                "predicted_clusters"
            ] = mpckmeans_dict_of_predicted_clusters

        if affinity_propagation:
            # Apply clustering with Affinity propagation

            start_time_affinity_propagation = time.time()
            affinity_propagation_dict_of_predicted_clusters = affinity_propagation_model.cluster(
                constraints_manager=constraints_manager, vectors=dict_of_vectors, nb_clusters=number_of_clusters
            )
            affinity_propagation_time = time.time() - start_time_affinity_propagation

            # Compute homogeneity, completeness and V-measure.

            dict_of_clustering_performances["affinity_propagation"][str(iteration)][
                "homogeneity"
            ] = metrics.homogeneity_score(
                list(dict_of_real_clusters.values()), list(affinity_propagation_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["affinity_propagation"][str(iteration)][
                "completeness"
            ] = metrics.completeness_score(
                list(dict_of_real_clusters.values()), list(affinity_propagation_dict_of_predicted_clusters.values())
            )

            dict_of_clustering_performances["affinity_propagation"][str(iteration)][
                "v_measure"
            ] = metrics.v_measure_score(
                list(dict_of_real_clusters.values()), list(affinity_propagation_dict_of_predicted_clusters.values())
            )

            # Store clustering time.

            dict_of_clustering_performances["affinity_propagation"][str(iteration)]["time"] = affinity_propagation_time

            # Store number of clusters.

            dict_of_clustering_performances["affinity_propagation"][str(iteration)][
                "nb_clusters"
            ] = affinity_propagation_model.number_of_clusters

            # Store predicted clusters.

            dict_of_clustering_performances["affinity_propagation"][str(iteration)][
                "predicted_clusters"
            ] = affinity_propagation_dict_of_predicted_clusters

        print("Iteration " + str(iteration) + "/" + str(number_of_iterations))

        current_time = time.time() - start_time

        hours = int(current_time / 3600)
        minuts = int((int(current_time) % 3600) / 60)
        seconds = int(current_time) % 60

        print("--- " + str(hours) + "h " + str(minuts) + "min " + str(seconds) + "s ---")

    return dict_of_clustering_performances


def plot_results(dict_of_clustering_performances,
                 algo_name,
                 constraints_increment=499,
                 plot_nb_clusters=False,
                 print_time=False
                 ):
    """
    Plots results of a performance measure over only one algorithm.
    Can display total clustering time.

    Args:
        dict_of_clustering_performances :  Dictionary with as keys the
         iteration numbers, and as values dictionaries containing the associated measured homogeneity,
          completeness, V-measuree, execution time, and number of clusters.
        algo_name : Name of the algorithm.
        constraints_increment : Number of constraints added at each iteration in performance measure process.
        plot_nb_clusters : Boolean to display number of clusters.
        print_time : Boolean to print total clustering time.

    """

    number_of_iterations = len(dict_of_clustering_performances)

    constraints_list = [constraints_increment * i for i in range(number_of_iterations)]

    homogeneity_list = [dict_of_clustering_performances[str(i)]["homogeneity"] for i in range(number_of_iterations)]
    completeness_list = [dict_of_clustering_performances[str(i)]["completeness"] for i in range(number_of_iterations)]
    v_measure_list = [dict_of_clustering_performances[str(i)]["v_measure"] for i in range(number_of_iterations)]
    number_of_clusters_list = [dict_of_clustering_performances[str(i)]["nb_clusters"] for i in range(number_of_iterations)]

    if print_time:
        model_total_time = 0
        for iteration in dict_of_clustering_performances:
            model_total_time += dict_of_clustering_performances[iteration]["time"]

        model_total_hours = int(model_total_time / 3600)
        model_total_minuts = int((int(model_total_time) % 3600) / 60)
        model_total_seconds = int(model_total_time) % 60

        print(
            "Total clusterings time for " + algo_name + " : "
            + str(model_total_hours)
            + " h, "
            + str(model_total_minuts)
            + " min, "
            + str(model_total_seconds)
            + " s"
        )

    nb_plots = 3

    if plot_nb_clusters:
        nb_plots += 1

    fig, axs = plt.subplots(nb_plots, 1, constrained_layout=True)

    ylim_min = min(homogeneity_list + completeness_list + v_measure_list) * 0.95
    ylim_max = max(homogeneity_list + completeness_list + v_measure_list) * 1.05

    axs[0].plot(constraints_list, homogeneity_list, "--")
    axs[0].set_xlabel("Number of constraints")
    axs[0].set_title("Homogeneity for " + algo_name)
    axs[0].set_ylabel("Homogeneity score")
    axs[0].set_ylim((ylim_min,ylim_max))

    axs[1].plot(constraints_list, completeness_list, "--")
    axs[1].set_xlabel("Number of constraints")
    axs[1].set_title("Completeness for " + algo_name)
    axs[1].set_ylabel("Completeness score")
    axs[1].set_ylim((ylim_min, ylim_max))

    axs[2].plot(constraints_list, v_measure_list, "--")
    axs[2].set_xlabel("Number of constraints")
    axs[2].set_title("V-measure for " + algo_name)
    axs[2].set_ylabel("V-measure score")
    axs[2].set_ylim((ylim_min, ylim_max))

    if plot_nb_clusters:
        axs[3].plot(constraints_list, number_of_clusters_list, "--")
        axs[3].set_xlabel("Number of constraints")
        axs[3].set_title("Number of clusters for " + algo_name)
        axs[3].set_ylabel("Number of clusters")

    plt.show()


def plot_all_results(
    dict_of_clustering_performances,
    title="Results for all algorithms",
    constraints_increment=7,
    plot_nb_clusters=False,
    kmeans=True,
    c_dbscan=True,
    mpckmeans=True,
    affinity_propagation=True,
):
    """
    Plots results of a performance measure over all the algorithms.
    Displays total clustering time for each algorithm.

    Args:
        dict_of_clustering_performances :  Dictionary that contains, for each algorithm, a dictionary with as keys the
         iteration numbers, and as values dictionaries containing the associated measured homogeneity,
          completeness, V-measuree, execution time, and number of clusters.
        title : Title of the plot.
        constraints_increment : Number of constraints added at each iteration in performance measure process.
        plot_nb_clusters : Boolean to display number of clusters or not.
        kmeans : Boolean to consider or not Kmeans.
        c_dbscan : Boolean to consider or not C-DBScan.
        mpckmeans : Boolean to consider or not MPCKmeans.
        affinity_propagation : Boolean to consider or not Affinity propagation.
    """

    # Compute total times

    dict_of_times = {model_name: [] for model_name in dict_of_clustering_performances.keys()}

    # Plot desired graphs and print total time

    nb_plots = 3

    if plot_nb_clusters:
        nb_plots += 1

    fig, axs = plt.subplots(nb_plots, 1, constrained_layout=True)

    number_of_iterations = len(dict_of_clustering_performances["kmeans"])

    constraints_list = [constraints_increment * i for i in range(number_of_iterations)]

    all_scores_list = []

    if kmeans:

        homogeneity_list_kmeans = [
            dict_of_clustering_performances["kmeans"][str(i)]["homogeneity"] for i in range(number_of_iterations)
        ]
        completeness_list_kmeans = [
            dict_of_clustering_performances["kmeans"][str(i)]["completeness"] for i in range(number_of_iterations)
        ]
        v_measure_list_kmeans = [
            dict_of_clustering_performances["kmeans"][str(i)]["v_measure"] for i in range(number_of_iterations)
        ]

        all_scores_list = all_scores_list + homogeneity_list_kmeans + completeness_list_kmeans + v_measure_list_kmeans

        nb_clusters_list_kmeans = [
            dict_of_clustering_performances["kmeans"][str(i)]["nb_clusters"] for i in range(number_of_iterations)
        ]

        axs[0].plot(constraints_list, homogeneity_list_kmeans, "-b", label="COP K-means")
        axs[1].plot(constraints_list, completeness_list_kmeans, "-b", label="COP K-means")
        axs[2].plot(constraints_list, v_measure_list_kmeans, "-b", label="COP K-means")

        if plot_nb_clusters:
            axs[3].plot(constraints_list, nb_clusters_list_kmeans, "-b", label="COP K-means")

        model_total_time = 0
        for iteration in dict_of_clustering_performances["kmeans"]:
            model_total_time += dict_of_clustering_performances["kmeans"][iteration]["time"]

        model_total_hours = int(model_total_time / 3600)
        model_total_minuts = int((int(model_total_time) % 3600) / 60)
        model_total_seconds = int(model_total_time) % 60

        print(
            "Total clusterings time for COP K-means : "
            + str(model_total_hours)
            + " h, "
            + str(model_total_minuts)
            + " min, "
            + str(model_total_seconds)
            + " s"
        )

    if c_dbscan:

        homogeneity_list_c_dbscan = [
            dict_of_clustering_performances["c_dbscan"][str(i)]["homogeneity"] for i in range(number_of_iterations)
        ]
        completeness_list_c_dbscan = [
            dict_of_clustering_performances["c_dbscan"][str(i)]["completeness"] for i in range(number_of_iterations)
        ]
        v_measure_list_c_dbscan = [
            dict_of_clustering_performances["c_dbscan"][str(i)]["v_measure"] for i in range(number_of_iterations)
        ]

        all_scores_list = all_scores_list + homogeneity_list_c_dbscan + completeness_list_c_dbscan +\
                          v_measure_list_c_dbscan

        nb_clusters_list_c_dbscan = [
            dict_of_clustering_performances["c_dbscan"][str(i)]["nb_clusters"] for i in range(number_of_iterations)
        ]

        axs[0].plot(constraints_list, homogeneity_list_c_dbscan, "-g", label="C-DBScan")
        axs[1].plot(constraints_list, completeness_list_c_dbscan, "-g", label="C-DBScan")
        axs[2].plot(constraints_list, v_measure_list_c_dbscan, "-g", label="C-DBScan")

        if plot_nb_clusters:
            axs[3].plot(constraints_list, nb_clusters_list_c_dbscan, "-g", label="C-DBScan")

        model_total_time = 0
        for iteration in dict_of_clustering_performances["c_dbscan"]:
            model_total_time += dict_of_clustering_performances["c_dbscan"][iteration]["time"]

        model_total_hours = int(model_total_time / 3600)
        model_total_minuts = int((int(model_total_time) % 3600) / 60)
        model_total_seconds = int(model_total_time) % 60

        print(
            "Total clusterings time for C-DBScan : "
            + str(model_total_hours)
            + " h, "
            + str(model_total_minuts)
            + " min, "
            + str(model_total_seconds)
            + " s"
        )

    if mpckmeans:

        homogeneity_list_mpckmeans = [
            dict_of_clustering_performances["mpckmeans"][str(i)]["homogeneity"] for i in range(number_of_iterations)
        ]
        completeness_list_mpckmeans = [
            dict_of_clustering_performances["mpckmeans"][str(i)]["completeness"] for i in range(number_of_iterations)
        ]
        v_measure_list_mpckmeans = [
            dict_of_clustering_performances["mpckmeans"][str(i)]["v_measure"] for i in range(number_of_iterations)
        ]

        all_scores_list = all_scores_list + homogeneity_list_mpckmeans + completeness_list_mpckmeans +\
                          v_measure_list_mpckmeans

        nb_clusters_list_mpckmeans = [
            dict_of_clustering_performances["mpckmeans"][str(i)]["nb_clusters"] for i in range(number_of_iterations)
        ]

        axs[0].plot(constraints_list, homogeneity_list_mpckmeans, "-r", label="MPCK-means")
        axs[1].plot(constraints_list, completeness_list_mpckmeans, "-r", label="MPCK-means")
        axs[2].plot(constraints_list, v_measure_list_mpckmeans, "-r", label="MPCK-means")

        if plot_nb_clusters:
            axs[3].plot(constraints_list, nb_clusters_list_mpckmeans, "-r", label="MPCK-means")

        model_total_time = 0
        for iteration in dict_of_clustering_performances["mpckmeans"]:
            model_total_time += dict_of_clustering_performances["mpckmeans"][iteration]["time"]

        model_total_hours = int(model_total_time / 3600)
        model_total_minuts = int((int(model_total_time) % 3600) / 60)
        model_total_seconds = int(model_total_time) % 60

        print(
            "Total clusterings time for MPCK-means : "
            + str(model_total_hours)
            + " h, "
            + str(model_total_minuts)
            + " min, "
            + str(model_total_seconds)
            + " s"
        )

    if affinity_propagation:
        homogeneity_list_affinity_propagation = [
            dict_of_clustering_performances["affinity_propagation"][str(i)]["homogeneity"]
            for i in range(number_of_iterations)
        ]
        completeness_list_affinity_propagation = [
            dict_of_clustering_performances["affinity_propagation"][str(i)]["completeness"]
            for i in range(number_of_iterations)
        ]
        v_measure_list_affinity_propagation = [
            dict_of_clustering_performances["affinity_propagation"][str(i)]["v_measure"]
            for i in range(number_of_iterations)
        ]

        all_scores_list = all_scores_list + homogeneity_list_affinity_propagation +\
                          completeness_list_affinity_propagation + v_measure_list_affinity_propagation

        nb_clusters_list_affinity_propagation = [
            dict_of_clustering_performances["affinity_propagation"][str(i)]["nb_clusters"]
            for i in range(number_of_iterations)
        ]

        axs[0].plot(constraints_list, homogeneity_list_affinity_propagation, "-y", label="Affinity Propagation")
        axs[1].plot(constraints_list, completeness_list_affinity_propagation, "-y", label="Affinity Propagation")
        axs[2].plot(constraints_list, v_measure_list_affinity_propagation, "-y", label="Affinity Propagation")

        if plot_nb_clusters:
            axs[3].plot(constraints_list, nb_clusters_list_affinity_propagation, "-y", label="Affinity Propagation")

        model_total_time = 0
        for iteration in dict_of_clustering_performances["affinity_propagation"]:
            model_total_time += dict_of_clustering_performances["affinity_propagation"][iteration]["time"]

        model_total_hours = int(model_total_time / 3600)
        model_total_minuts = int((int(model_total_time) % 3600) / 60)
        model_total_seconds = int(model_total_time) % 60

        print(
            "Total clusterings time for Affinity Propagation : "
            + str(model_total_hours)
            + " h, "
            + str(model_total_minuts)
            + " min, "
            + str(model_total_seconds)
            + " s"
        )

    ylim_min = min(all_scores_list) * 0.95
    ylim_max = max(all_scores_list) * 1.05

    axs[0].legend(loc="lower right")
    axs[0].set_xlabel("Number of constraints")
    axs[0].set_title("Homogeneity")
    axs[0].set_ylabel("Homogeneity score")
    axs[0].set_ylim((ylim_min, ylim_max))

    axs[1].legend(loc="lower right")
    axs[1].set_xlabel("Number of constraints")
    axs[1].set_title("Completeness")
    axs[1].set_ylabel("Completeness score")
    axs[1].set_ylim((ylim_min, ylim_max))

    axs[2].legend(loc="lower right")
    axs[2].set_xlabel("Number of constraints")
    axs[2].set_title("V-measure")
    axs[2].set_ylabel("V-measure score")
    axs[2].set_ylim((ylim_min, ylim_max))

    if plot_nb_clusters:

        axs[3].legend(loc="lower right")
        axs[3].set_xlabel("Number of constraints")
        axs[3].set_title("Number of clusters")
        axs[3].set_ylabel("Number of clusters")

    fig.suptitle(title)

    plt.show()
