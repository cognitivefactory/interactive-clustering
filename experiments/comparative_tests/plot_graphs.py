from tests.comparative_tests.utils import load_results, plot_all_results, plot_results

# For all the algorithms at the same time over a 60-entries reduced dataset

dict_of_clustering_performances = load_results("./measures_results/kmeans_c_dbscan_mpckmeans_affinity_size_60.json")

plot_all_results(
    dict_of_clustering_performances, title="Results with a 60-entries reduced dataset", plot_nb_clusters=True
)

plot_results(dict_of_clustering_performances["kmeans"], algo_name="COP K-means", constraints_increment=7)
plot_results(dict_of_clustering_performances["c_dbscan"], algo_name="C-DBScan", constraints_increment=7)
plot_results(dict_of_clustering_performances["mpckmeans"], algo_name="MPCK-means", constraints_increment=7)
plot_results(
    dict_of_clustering_performances["affinity_propagation"], algo_name="Affinity Propagation", constraints_increment=7
)

# For K-means, C-DBScan and Affinity Propagation at the same time over full dataset

dict_of_clustering_performances = load_results("./measures_results/kmeans_c_dbscan_affinity_size_500.json")

plot_all_results(
    dict_of_clustering_performances,
    title="Results with full dataset",
    constraints_increment=499,
    plot_nb_clusters=True,
    mpckmeans=False,
)

plot_results(dict_of_clustering_performances["kmeans"], algo_name="COP K-means")
plot_results(dict_of_clustering_performances["c_dbscan"], algo_name="C-DBScan")
plot_results(dict_of_clustering_performances["affinity_propagation"], algo_name="Affinity Propagation")
