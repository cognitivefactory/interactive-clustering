This folder contains files for measuring of algorithms' performances.

The idea of the measure consists in performing clustering by adding iteratively some Must-link/Cannot-link constraints randomly.
Homogeneity, completeness, V-measure, number of clusters and clustering time are measured at each iteration.

All the functions for these measures are implemented in the Python script *utils.py*.

The measures can be run with the Python notebook *performances.ipynb*.
(You may have to change the working directories in the firsts cells. Otherwise, each cell can be run individually.)

Results are saved in *.json* files, in the directory */measures_result* (some examples are already available).

Results can be plotted with the Python notebook *performances.ipynb*, but also by running the Python script *plot_graphs.py*