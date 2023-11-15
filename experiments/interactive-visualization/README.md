# Interactive visualization

This script create an interactive web visualization of the clustering performed during comparative tests.

## How ?

The script first loads in a pickled file of vectors, performs a t-SNE transformation to reduce the dimensions of the vectors to 3, and then loads in a json file of clustering results.

The script then creates a Dash app with a dropdown menu to select an algorithm (C-DBScan, MPCKmeans, Affinity Propagation, or Kmeans) and another dropdown menu to select an iteration number. A 3D scatter plot of the data is displayed, with the points colored according to their predicted cluster, and the hover name displaying the name of the vector.

The figure updates when the user selects a different algorithm or iteration number. If no algorithm or iteration is selected, the app will not update.

A sample of clustering results is provided in the `data` folder.
