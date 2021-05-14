# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.interactive_clustering.clustering.spectral
* Description:  Implementation of constrained spectral clustering algorithms.
* Author:       Erwan Schild
* Created:      17/03/2021
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORTS PYTHON DEPENDENCIES
# ==============================================================================

# Dependency needed to shuffle data and set random seed.

# Python code typing (mypy).
from typing import Dict, List, Optional, Union

# Dependencies needed handle float and matrix.
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

# Dependencies needed to use classical clustering algorithms.
from sklearn.cluster import SpectralClustering

# Dependencies needed to compute similary matrix.
from sklearn.metrics import pairwise_kernels

# The needed clustering abstract class and utilities methods.
from cognitivefactory.interactive_clustering.clustering.abstract import (
    AbstractConstrainedClustering,
    rename_clusters_by_order,
)

# Dependency needed to manage constraints.
from cognitivefactory.interactive_clustering.constraints.abstract import AbstractConstraintsManager
from cognitivefactory.interactive_clustering.utils import checking

##from scipy.linalg import sqrtm

# Dependencies needed to solve semidefine programming.
##import cvxpy as cp #TODO Replace by cvxopt

# Dependencies needed to solve eigen value computation.
##from scipy.sparse.csgraph import laplacian as csgraph_laplacian
##from scipy.sparse.linalg import eigsh

##from sklearn.cluster import KMeans


# ==============================================================================
# SPECTRAL CONSTRAINED CLUSTERING
# ==============================================================================
class SpectralConstrainedClustering(AbstractConstrainedClustering):
    """
    This class implements the spectral constrained clustering.
    It inherits from `AbstractConstrainedClustering`.

    References:
        - Spectral Clustering: `Ng, A. Y., M. I. Jordan, et Y.Weiss (2002). On Spectral Clustering: Analysis and an algorithm. In T. G. Dietterich, S. Becker, et Z. Ghahramani (Eds.), Advances in Neural Information Processing Systems 14. MIT Press.`
        - Constrained _'SPEC'_ Spectral Clustering: `Kamvar, S. D., D. Klein, et C. D. Manning (2003). Spectral Learning. Proceedings of the international joint conference on artificial intelligence, 561–566.`
    """

    # ==============================================================================
    # INITIALIZATION
    # ==============================================================================
    def __init__(
        self, model: str = "SPEC", nb_components: Optional[int] = None, random_seed: Optional[int] = None, **kargs
    ) -> None:
        """
        The constructor for Spectral Constrainted Clustering class.

        Args:
            model (str, optional): The spectral clustering model to use. Available spectral models are `"SPEC"` and `"CCSR"`. Defaults to `"SPEC"`.
            nb_components (Optional[int], optional): The number of eigenvectors to compute in the spectral clustering. If `None`, set the number of components to the number of clusters. Defaults to `None`.
            random_seed (Optional[int], optional): The random seed to use to redo the same clustering. Defaults to `None`.
            **kargs (dict): Other parameters that can be used in the instantiation.

        Raises:
            ValueError: if some parameters are incorrectly set.
        """

        # Store `self.model`.
        if model != "SPEC":  # TODO use `not in {"SPEC"}`. # TODO `"CCSR"` to add after correction.
            raise ValueError("The `model` '" + str(model) + "' is not implemented.")
        self.model: str = model

        # Store `self.nb_components`.
        if (nb_components is not None) and (nb_components < 2):
            raise ValueError(
                "The `nb_components` '" + str(nb_components) + "' must be `None` or greater than or equal to 2."
            )
        self.nb_components: Optional[int] = nb_components

        # Store `self.random_seed`.
        self.random_seed: Optional[int] = random_seed

        # Store `self.kargs` for kmeans clustering.
        self.kargs = kargs

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters: Optional[Dict[str, int]] = None

    # ==============================================================================
    # MAIN - CLUSTER DATA
    # ==============================================================================
    def cluster(
        self,
        vectors: Dict[str, Union[ndarray, csr_matrix]],
        nb_clusters: int,
        constraints_manager: Optional[AbstractConstraintsManager] = None,
        verbose: bool = False,
        **kargs,
    ) -> Dict[str, int]:
        """
        The main method used to cluster data with the Spectral model.

        Args:
            vectors (Dict[str,Union[ndarray,csr_matrix]]): The representation of data vectors. The keys of the dictionary represents the data IDs. This keys have to refer to the list of data IDs managed by the `constraints_manager` (if `constraints_manager` is set). The value of the dictionary represent the vector of each data. Vectors can be dense (`numpy.ndarray`) or sparse (`scipy.sparse.csr_matrix`).
            nb_clusters (int): The number of clusters to compute. #TODO Set defaults to None with elbow method or other method ?
            constraints_manager (Optional[AbstractConstraintsManager], optional): A constraints manager over data IDs that will force clustering to respect some conditions during computation. The list of data IDs managed by `constraints_manager` has to refer to `vectors` keys. If `None`, no constraint are used during the clustering. Defaults to `None`.
            verbose (bool, optional): Enable verbose output. Defaults to `False`.
            **kargs (dict): Other parameters that can be used in the clustering.

        Raises:
            ValueError: if `vectors` and `constraints_manager` are incompatible, or if some parameters are incorrectly set.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """

        ###
        ### GET PARAMETERS
        ###

        # Get `list_of_data_IDs`.
        if not isinstance(vectors, dict):
            raise ValueError("The `vectors` parameter has to be a `dict` type.")
        self.list_of_data_IDs: List[str] = sorted(vectors.keys())
        if not isinstance(self.list_of_data_IDs, list) or not all(
            isinstance(element, str) for element in self.list_of_data_IDs
        ):
            raise ValueError("The `list_of_data_IDs` variable has to be a `list` type.")

        # Check `constraints_manager`.
        self.constraints_manager: AbstractConstraintsManager = checking.check_constraints_manager(
            list_of_data_IDs=self.list_of_data_IDs,
            constraints_manager=constraints_manager,
        )

        # Check `vectors`.
        self.vectors: Dict[str, Union[ndarray, csr_matrix]] = checking.check_vectors(
            list_of_data_IDs=self.list_of_data_IDs,
            vectors=vectors,
        )

        # Store `self.nb_clusters`.
        if nb_clusters < 2:
            raise ValueError("The `nb_clusters` '" + str(nb_clusters) + "' must be greater than or equal to 2.")
        self.nb_clusters = nb_clusters

        # Define `self.current_nb_components`.
        self.current_nb_components: int = (
            self.nb_components
            if ((self.nb_components is not None) and (self.nb_clusters < self.nb_components))
            else self.nb_clusters
        )

        # Compute `self.pairwise_similarity_matrix`.
        self.pairwise_similarity_matrix = csr_matrix(
            [
                [
                    pairwise_kernels(X=self.vectors[data_ID1], Y=self.vectors[data_ID2], metric="rbf")[0][0].astype(
                        np.float64
                    )
                    for data_ID2 in self.list_of_data_IDs
                ]
                for data_ID1 in self.list_of_data_IDs
            ]
        )

        ###
        ### RUN SPECTRAL CONSTRAINED CLUSTERING
        ###

        # Initialize `self.dict_of_predicted_clusters`.
        self.dict_of_predicted_clusters = None

        # Case of `"CCSR"` spectral clustering.
        # TODO Don't work.
        ##if self.model == "CCSR":
        ##    self.dict_of_predicted_clusters = self.clustering_spectral_model_CCSR(verbose=verbose)

        # Case of `"SPEC"` spectral clustering.
        ##### DEFAULTS : if self.model=="SPEC":
        self.dict_of_predicted_clusters = self.clustering_spectral_model_SPEC(verbose=verbose)

        ###
        ### RETURN PREDICTED CLUSTERS
        ###

        return self.dict_of_predicted_clusters

    # ==============================================================================
    # IMPLEMENTATION - SPEC SPECTRAL CLUSTERING
    # ==============================================================================
    def clustering_spectral_model_SPEC(
        self,
        verbose: bool = False,
    ) -> Dict[str, int]:
        """
        Implementation of a simple Spectral clustering algorithm, based affinity matrix modifications.

        References :
            - Constrained _'SPEC'_ Spectral Clustering: `Kamvar, S. D., D. Klein, et C. D. Manning (2003). Spectral Learning. Proceedings of the international joint conference on artificial intelligence, 561–566.`

        Args:
            verbose (bool, optional): Enable verbose output. Default is `False`.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        """

        ###
        ### MODIFY CONSTRAINTS MATRIX WITH CONSTRAINTS
        ###

        # Modify the similarity over data IDs.
        for ID1, data_ID1 in enumerate(self.list_of_data_IDs):
            for ID2, data_ID2 in enumerate(self.list_of_data_IDs):

                # Symetry is already handled in next instructions.
                if ID1 > ID2:
                    continue

                # For each `"MUST_LINK"` constraint, set the similarity to 1.0.
                if (
                    self.constraints_manager.get_inferred_constraint(
                        data_ID1=data_ID1,
                        data_ID2=data_ID2,
                    )
                    == "MUST_LINK"
                ):
                    self.pairwise_similarity_matrix[ID1, ID2] = 1.0
                    self.pairwise_similarity_matrix[ID2, ID1] = 1.0

                # For each `"CANNOT_LINK"` constraint, set the similarity to 0.0.
                elif (
                    self.constraints_manager.get_inferred_constraint(
                        data_ID1=data_ID1,
                        data_ID2=data_ID2,
                    )
                    == "CANNOT_LINK"
                ):
                    self.pairwise_similarity_matrix[ID1, ID2] = 0.0
                    self.pairwise_similarity_matrix[ID2, ID1] = 0.0

        ###
        ### RUN SPECTRAL CONSTRAINED CLUSTERING
        ###     | Define laplacian matrix
        ###     | Compute eigen vectors
        ###     | Cluster eigen vectors
        ###     | Return labels based on eigen vectors clustering
        ###

        # Initialize spectral clustering model.
        self.clustering_model = SpectralClustering(
            n_clusters=self.nb_clusters,
            # n_components=self.current_nb_components, #TODO Add if `scikit-learn>=0.24.1`
            affinity="precomputed",
            random_state=self.random_seed,
            **self.kargs,
        )

        # Run spectral clustering model.
        self.clustering_model.fit_predict(X=self.pairwise_similarity_matrix)

        # Get prediction of spectral clustering model.
        list_of_clusters: List[int] = self.clustering_model.labels_.tolist()

        # Define the dictionary of predicted clusters.
        predicted_clusters: Dict[str, int] = {
            data_ID: list_of_clusters[ID] for ID, data_ID in enumerate(self.list_of_data_IDs)
        }

        # Rename cluster IDs by order.
        predicted_clusters = rename_clusters_by_order(clusters=predicted_clusters)

        # Return predicted clusters
        return predicted_clusters

    # ==============================================================================
    # IMPLEMENTATION - CCSR SPECTRAL CLUSTERING
    # ==============================================================================
    """ #TODO : do not work. check if 1) wrong implementation ? 2) cvxopt better ?
    def clustering_spectral_model_CCSR(
        self,
        verbose: bool = False,
    ) -> Dict[str, int]:
        \"""
        Implementation of Constrained Clustering with Spectral Regularization algorithm, based on spectral semidefinite programming interpretation.
        - Source : `Li, Z., Liu, J., & Tang, X. (2009). Constrained clustering via spectral regularization. 2009 IEEE Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2009.5206852`
        - MATLAB Implementation : https://icube-forge.unistra.fr/lampert/TSCC/-/tree/master/methods/Li09
        - SOLVERS comparison : https://pypi.org/project/PICOS/
        - CVXPY solver : https://www.cvxpy.org/index.html

        Args:
            verbose (bool, optional): Enable verbose output. Defaults to `False`.

        Returns:
            Dict[str,int]: A dictionary that contains the predicted cluster for each data ID.
        \"""

        ###
        ### COMPUTE NORMALIZED LAPLACIAN
        ###

        # Compute the normalized Laplacian.
        normalized_laplacian, diagonal = csgraph_laplacian(
            csgraph=self.pairwise_similarity_matrix,
            normed=True,
            return_diag=True
        )

        ###
        ### COMPUTE EIGENVECTORS OF LAPLACIAN
        ###

        random.seed(self.random_seed)
        v0 = np.random.rand(min(normalized_laplacian.shape))

        # Compute the largest eigen vectors.
        eigen_values, eigen_vectors = eigsh(
            A=normalized_laplacian,
            which="SM",
            k=self.current_nb_components + 1,
            v0=v0,
        )

        # Ignore eigenvector of eigenvalue 0.
        eigen_values = eigen_values[1:]
        eigen_vectors = eigen_vectors.T[1:]

        if verbose:
            print("EIGENVALUES / EIGENVECTORS")

            for k in range(len(eigen_values)):
                print("    ", "=============")
                print("    ", "ID :         ", k)
                print("    ", "VAL :        ", eigen_values[k])
                print("    ", "VEC :        ", eigen_vectors[k])

        ###
        ### FORMALIZE SEMIDEFINITE PROBLEM
        ###

        ## Problem ::
        ## Cost function to minimize : L(z) = 1/2 * z.T * B * z + b.T * z + c
        ## z : variable to find with the SDP problem
        ## z = vec(M)
        ## M >> 0 (semi definite positive), i.e. M = M.T, M.shape=(nb_components, nb_components)
        ## B = sum ( s_ij s_ij.T )
        # for ij in MUST_LINK or i,j in CANNOT_LINK
        ## b = -2 * sum ( t_ij * s_ij )
        # for ij in MUST_LINK or i,j in CANNOT_LINK
        ## c = sum ( t_ij^2 )
        # for ij in MUST_LINK or i,j in CANNOT_LINK
        ## s_ij = vec( eigen_vector_i.T * eigen_vector_j )
        ## t_ij = 1 if MUST_LINK(i,j), 0 if CANNOT_LINK(i,j)

        # Initialization of SDP variables.
        B = np.zeros((self.current_nb_components ** 2, self.current_nb_components ** 2))
        b = np.zeros((self.current_nb_components ** 2, 1))
        c = 0

        for ID1, data_ID1 in enumerate(self.list_of_data_IDs):
            for ID2, data_ID2 in enumerate(self.list_of_data_IDs):

                # Get eigenvectors.
                eigen_vector_i = np.atleast_2d(eigen_vectors.T[ID1])
                eigen_vector_j = np.atleast_2d(eigen_vectors.T[ID2])

                # Compute eigenvectors similarity.
                U = eigen_vector_j.T @ eigen_vector_i
                s = np.atleast_2d(U.ravel())


                # For each `"MUST_LINK"` constraint, ....
                if self.constraints_manager.get_inferred_constraint(
                    data_ID1=data_ID1,
                    data_ID2=data_ID2,
                ) == "MUST_LINK":

                    # Add the value to SDP variables.
                    B += s.T * s
                    b += - 1 * s.T
                    c += 1 * 1

                # For each `"CANNOT_LINK"` constraint, ....
                if self.constraints_manager.get_inferred_constraint(
                    data_ID1=data_ID1,
                    data_ID2=data_ID2,
                ) == "CANNOT_LINK":

                    # Add the value to SDP variables.
                    B += s.T * s
                    b += - 0 * s.T
                    c += 0 * 0

        ###
        ### SOLVE SEMIDEFINITE PROBLEM
        ###

        # Create a symetric matrix variable.
        M = cp.Variable((self.current_nb_components, self.current_nb_components))

        ### Define constraints.
        SDP_constraints = []

        # The solution must be positive semidefinite.
        SDP_constraints += [M >> 0]

        # Define cost function to minimize : `( 1/2 * z.T * B * z + b.T * z + c )`.
        self.SDP_problem = cp.Problem(
            cp.Minimize(
                cp.quad_form(cp.atoms.affine.vec.vec(M), B)  # `1/2 * z.T * B * z`.
                + b.T @ cp.atoms.affine.vec.vec(M)  # `b.T * z`.
                + c  # c
            ),
            SDP_constraints,
        )

        # Solve the SDP problem.
        self.SDP_problem.solve(solver="MOSEK")
        if verbose:
            print("SEMIDEFINITE PROBLEM")
            print("    ", "STATUS", ":", self.SDP_problem.status)
            print("    ", "COST FUNCTION VALUE", ":", self.SDP_problem.value)

        ###
        ### CLUSTER EIGEN VECTORS
        ###

        # Define square root of M, and force sqrtM to be symetric.
        sqrtM = sqrtm(M.value).real
        sqrtM = (sqrtM + sqrtM.T) / 2

        # Compute new embeddings for spectral clustering.
        new_vectors_to_clusters = eigen_vectors.T @ sqrtM

        # Initialize kmeans klustering model.
        self.clustering_model = KMeans(
            n_clusters=self.nb_clusters,
            max_iter=10000,
            random_state=self.random_seed,
        )

        # Run kmeans clustering model.
        self.clustering_model.fit_predict(
            X=new_vectors_to_clusters
        )

        # Get prediction of kmeans clustering model.
        list_of_clusters: List[int] = self.clustering_model.labels_.tolist()

        # Define the dictionary of predicted clusters.
        predicted_clusters: Dict[str, int] = {
            data_ID: list_of_clusters[ID]
            for ID, data_ID in enumerate(self.list_of_data_IDs)
        }

        ###
        ### RENAME CLUSTERS BY ORDER
        ###

        # Define a map to be able to rename cluster IDs.
        mapping_of_old_ID_to_new_ID: Dict[int, int] = {}
        new_ID: int = 0
        for data_ID in self.list_of_data_IDs:
            if predicted_clusters[data_ID] not in mapping_of_old_ID_to_new_ID.keys():
                mapping_of_old_ID_to_new_ID[predicted_clusters[data_ID]] = new_ID
                new_ID += 1

        # Rename cluster IDs.
        predicted_clusters = {
            data_ID: mapping_of_old_ID_to_new_ID[predicted_clusters[data_ID]]
            for data_ID in self.list_of_data_IDs
        }

        # Return predicted clusters
        return predicted_clusters
    """
