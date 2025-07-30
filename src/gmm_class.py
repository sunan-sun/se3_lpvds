import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from .util.quat_tools import *
from .util.plot_tools import *


def adjust_cov(cov, tot_scale_fact_pos=2,  tot_scale_fact_ori=1.2, rel_scale_fact=0.15):
    """ Adjusts the eigenvalues of a covariance matrix to ensure minimum spread.

    Separately adjusts the position (first 3 dims) and orientation (last 4 dims)
    parts of the covariance matrix. If the ratio between certain eigenvalues is below
    `rel_scale_fact`, eigenvalues are increased to meet this minimum ratio, ensuring
    the Gaussian component is not overly flattened in certain directions. The overall
    scale is then adjusted by `tot_scale_fact_pos` and `tot_scale_fact_ori`.

    Args:
        cov (np.ndarray): The 7x7 covariance matrix to adjust.
        tot_scale_fact_pos (float): Factor to scale the position eigenvalues.
        tot_scale_fact_ori (float): Factor to scale the orientation eigenvalues.
        rel_scale_fact (float): Minimum allowed ratio between certain sorted eigenvalues.

    Returns:
        np.ndarray: The adjusted 7x7 covariance matrix.
    """

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    eigenvalues_pos = eigenvalues[: 3]
    idxs_pos = eigenvalues_pos.argsort()
    inverse_idxs_pos = np.zeros((idxs_pos.shape[0]), dtype=int)
    for index, element in enumerate(idxs_pos):
        inverse_idxs_pos[element] = index

    eigenvalues_sorted_pos  = np.sort(eigenvalues_pos)
    cov_ratio = eigenvalues_sorted_pos[1]/eigenvalues_sorted_pos[2]
    if cov_ratio < rel_scale_fact:
        lambda_3_pos = eigenvalues_sorted_pos[2]
        lambda_2_pos = eigenvalues_sorted_pos[1] + lambda_3_pos * (rel_scale_fact - cov_ratio)
        lambda_1_pos = eigenvalues_sorted_pos[0] + lambda_3_pos * (rel_scale_fact - cov_ratio)

        lambdas_pos = np.array([lambda_1_pos, lambda_2_pos, lambda_3_pos])
        L_pos = np.diag(lambdas_pos[inverse_idxs_pos]) * tot_scale_fact_pos
    else:
        L_pos = np.diag(eigenvalues_pos) * tot_scale_fact_pos


    eigenvalues_ori = eigenvalues[3: ]
    idxs_ori = eigenvalues_ori.argsort()
    inverse_idxs_ori = np.zeros((idxs_ori.shape[0]), dtype=int)
    for index, element in enumerate(idxs_ori):
        inverse_idxs_ori[element] = index

    eigenvalues_sorted_ori  = np.sort(eigenvalues_ori)
    cov_ratio = eigenvalues_sorted_ori[2]/eigenvalues_sorted_ori[3]
    if cov_ratio < rel_scale_fact:
        lambda_4_ori = eigenvalues_sorted_ori[3]
        lambda_3_ori = eigenvalues_sorted_ori[2] + lambda_4_ori * (rel_scale_fact - cov_ratio)
        lambda_2_ori = eigenvalues_sorted_ori[1] + lambda_4_ori * (rel_scale_fact - cov_ratio)
        lambda_1_ori = eigenvalues_sorted_ori[0] + lambda_4_ori * (rel_scale_fact - cov_ratio)

        lambdas_ori = np.array([lambda_1_ori, lambda_2_ori, lambda_3_ori, lambda_4_ori])
        L_ori = np.diag(lambdas_ori[inverse_idxs_ori]) * tot_scale_fact_ori
    else:
        L_ori = np.diag(eigenvalues_ori) * tot_scale_fact_ori


    L = np.zeros((7, 7))
    L[:3, :3] = L_pos
    L[3:, 3:] = L_ori

    Sigma = eigenvectors @ L @ eigenvectors.T

    return Sigma




class gmm_class:
    """ Implements a Gaussian Mixture Model (GMM) tailored for SE(3) data (position + orientation).

    Uses Bayesian GMM for clustering and handles the dual cover representation of quaternions.
    Orientation is represented in the tangent space (log map) relative to a reference orientation
    (typically the attractor) for GMM fitting.
    """
    def __init__(self, p_in:np.ndarray, q_in:list, q_att:R, K_init:int):
        """
        Initializes the GMM class with input SE(3) data.

        Parameters:
        ----------
            p_in (np.ndarray):      [M, 3] NumPy array of POSITION INPUT.
            q_in (list):            M-length list of Rotation objects for ORIENTATION INPUT.
            q_att (Rotation):       Single Rotation object for the reference ORIENTATION (ATTRACTOR).
            K_init (int):           Initial maximum number of Gaussian components for Bayesian GMM.
        """

        # Standardize quaternion signs (ensure scalar part w is negative)
        standardized_q_in = []
        for q in q_in:
            quat = q.as_quat()
            if quat[3] > 0:
                standardized_q_in.append(R.from_quat(-quat))
            else:
                standardized_q_in.append(q)

        q_att_quat = q_att.as_quat()
        if q_att_quat[3] > 0:
            q_att = R.from_quat(-q_att_quat)

        # store parameters
        self.p_in     = p_in
        self.q_in     = standardized_q_in
        self.q_att    = q_att
        self.K_init   = K_init

        self.M = len(q_in)
        self.N = 7 # Dimension (3 for position, 4 for tangent space orientation)

        # form concatenated state in tangent space for GMM fitting
        self.pq_in    = np.hstack((self.p_in, riem_log(self.q_att, self.q_in)))




    def fit(self):
        """
        Fits the Bayesian GMM to the input data.

        1. Fits a BayesianGaussianMixture to the concatenated position and
           tangent-space orientation data (self.pq_in).
        2. Predicts component assignments for each data point.
        3. Rearranges components to remove unused ones and ensure contiguous indexing.
        4. Extracts Gaussian parameters (Prior, Mu, Sigma) for each component and its dual.
        5. Calculates initial responsibilities (gamma) considering the dual cover.

        Returns:
        -------
            np.ndarray: [K, M] array of responsibilities (posterior probabilities)
                        for the primary Gaussian components (excluding duals).
        """

        self.gmm = BayesianGaussianMixture(n_components=self.K_init, n_init=1).fit(self.pq_in)
        assignment_arr = self.gmm.predict(self.pq_in)

        self._rearrange_array(assignment_arr)
        self._extract_gaussian()

        dual_gamma = self.logProb(self.p_in, self.q_in) # 2K by M

        return dual_gamma[:self.K, :] # K by M; always remain the first half
    

    def sample(self):
        pq = self.gmm.sample(n_samples=1)[0].flatten()
        p = pq[:3]
        q = pq[3:]
        # q = riem_exp(R.from_quat(-self.q_att.as_quat()), np.expand_dims(q, axis=0))
        q = riem_exp(self.q_att, np.expand_dims(q, axis=0))
        return p, q.flatten()



    def _rearrange_array(self, assignment_arr):
        """ Removes empty components and re-indexes assignments to be contiguous.

        After fitting Bayesian GMM, some components might be unused (zero weight).
        This function removes those components and updates the assignment array
        so that component indices are contiguous from 0 to K-1.

        Args:
        -----
            assignment_arr (np.ndarray): [M,] array of component assignments from GMM fit.

        Sets:
        -----
            self.K (int): The actual number of non-empty components.
            self.assignment_arr (np.ndarray): The updated [M,] assignment array.
        """
        rearrange_list = []
        new_assignment_arr = np.zeros_like(assignment_arr)
        for idx, entry in enumerate(assignment_arr):
            if not rearrange_list:
                rearrange_list.append(entry)
            if entry not in rearrange_list:
                rearrange_list.append(entry)
                new_assignment_arr[idx] = len(rearrange_list) - 1
            else:
                new_assignment_arr[idx] = rearrange_list.index(entry)   
        
        self.K = len(rearrange_list)
        self.assignment_arr = new_assignment_arr




    def _extract_gaussian(self):
        """
        Extracts Gaussian parameters (Prior, Mu, Sigma) for each component.

        Calculates the mean position, mean orientation (using quaternion averaging),
        prior probability, and covariance matrix for each component based on the
        assigned data points. Also calculates parameters for the dual cover
        representation of each orientation component. Covariances are adjusted
        using `adjust_cov`.

        Sets:
        -----
            self.Prior (list): [2*K] list of prior probabilities.
            self.Mu (list):    [2*K] list of tuples, each (mean_pos: np.ndarray[3,], mean_ori: R).
            self.Sigma (list): [2*K] list of [N, N] covariance matrices.
            self.gaussian_list (list): [K] list of dicts for primary components.
            self.dual_gaussian_list (list): [K] list of dicts for dual components.
        """

        assignment_arr = self.assignment_arr

        Prior   = [0] * (2 * self.K)
        Mu      = [(np.zeros((3, )), R.identity())] * (2 * self.K)
        Sigma   = [np.zeros((self.N, self.N), dtype=np.float32)] * (2 * self.K)

        gaussian_list = [] 
        dual_gaussian_list = []
        for k in range(self.K):
            q_k      = [q for index, q in enumerate(self.q_in) if assignment_arr[index]==k] 
            q_k_mean = quat_mean(q_k)

            p_k      = [p for index, p in enumerate(self.p_in) if assignment_arr[index]==k]
            p_k_mean = np.mean(np.array(p_k), axis=0)

            q_diff = riem_log(q_k_mean, q_k) 
            p_diff = p_k - p_k_mean
            pq_diff = np.hstack((p_diff, q_diff))

            Prior[k]  = len(q_k)/ (2 * self.M)
            Mu[k]     = (p_k_mean, q_k_mean)
            Sigma_k  = pq_diff.T @ pq_diff / (len(q_k)-1)  + 10E-6 * np.eye(self.N)
            # Sigma_k  = adjust_cov(Sigma_k)
            Sigma[k] = Sigma_k

            gaussian_list.append(
                {   
                    "prior" : Prior[k],
                    "mu"    : Mu[k],
                    "sigma" : Sigma[k],
                    "rv"    : multivariate_normal(np.hstack((Mu[k][0], np.zeros(4))), Sigma[k], allow_singular=True)
                }
            )


            q_k_dual  = [R.from_quat(-q.as_quat()) for q in q_k]
            q_k_mean_dual     = R.from_quat(-q_k_mean.as_quat())

            q_diff_dual = riem_log(q_k_mean_dual, q_k_dual)
            pq_diff_dual = np.hstack((p_diff, q_diff_dual))

            Prior[self.K + k] = Prior[k]
            Mu[self.K + k]     = (p_k_mean, q_k_mean_dual)
            Sigma_k_dual = pq_diff_dual.T @ pq_diff_dual / (len(q_k_dual)-1)  + 10E-6 * np.eye(self.N)
            # Sigma_k_dual  = adjust_cov(Sigma_k_dual)
            Sigma[self.K+k]  = Sigma_k_dual

            dual_gaussian_list.append(
                {   
                    "prior" : Prior[self.K + k],
                    "mu"    : Mu[self.K + k],
                    "sigma" : Sigma[self.K+k],
                    "rv"    : multivariate_normal(np.hstack((Mu[self.K + k][0], np.zeros(4))), Sigma[self.K + k], allow_singular=True)
                }
            )


        self.gaussian_list = gaussian_list
        self.dual_gaussian_list = dual_gaussian_list


        self.Prior  = Prior
        self.Mu     = Mu
        self.Sigma  = Sigma




    def logProb(self, p_in, q_in):
        """ Computes the posterior probabilities (responsibilities) for given data points.

        Calculates the probability of each data point (p_in, q_in) belonging to each
        Gaussian component (including duals) based on the learned GMM parameters.
        Uses the log-sum-exp trick for numerical stability.

        Args:
        -----
            p_in (np.ndarray): [M, 3] array of positions.
            q_in (list/Rotation): List of M Rotation objects or a single Rotation object
                                  representing orientations.

        Returns:
        -------
            np.ndarray: [2*K, M] array of posterior probabilities (responsibilities).
        """
        # Ensure q_in is always a list, even if a single Rotation object is passed
        if isinstance(q_in, R):
            q_in_list = [q_in]
            num_points = 1
        else:
            q_in_list = q_in
            num_points = len(q_in_list)

        # Ensure p_in has the correct shape
        if p_in.ndim == 1:
            p_in_arr = p_in.reshape(1, -1)
        else:
            p_in_arr = p_in

        if p_in_arr.shape[0] != num_points:
            raise ValueError("Number of positions and orientations must match.")

        logProb = np.zeros((2 * self.K, num_points))

        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.gaussian_list[k].values())

            # Project orientations into tangent space of the component mean orientation
            q_k_tangent = riem_log(mu_k[1], q_in_list)
            # Concatenate position and tangent-space orientation
            pq_k = np.hstack((p_in_arr, q_k_tangent))

            logProb[k, :] = np.log(prior_k) + normal_k.logpdf(pq_k)


        for k in range(self.K):
            prior_k, mu_k, _, normal_k = tuple(self.dual_gaussian_list[k].values())

            # Project orientations into tangent space of the component mean orientation (dual)
            q_k_tangent = riem_log(mu_k[1], q_in_list)
            # Concatenate position and tangent-space orientation
            pq_k = np.hstack((p_in_arr, q_k_tangent))

            logProb[k+self.K, :] = np.log(prior_k) + normal_k.logpdf(pq_k)


        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (2 * self.K, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

        return postProb
    



'''
def adjust_cov_pos(cov, tot_scale_fact=2, rel_scale_fact=0.15):
    
    cov_pos = cov[:3, :3]

    eigenvalues, eigenvectors = np.linalg.eig(cov_pos)

    idxs = eigenvalues.argsort()
    inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
    for index, element in enumerate(idxs):
        inverse_idxs[element] = index

    eigenvalues_sorted  = np.sort(eigenvalues)
    cov_ratio = eigenvalues_sorted[1]/eigenvalues_sorted[2]
    if cov_ratio < rel_scale_fact:
        lambda_3 = eigenvalues_sorted[2]
        lambda_2 = eigenvalues_sorted[1] + lambda_3 * (rel_scale_fact - cov_ratio)
        lambda_1 = eigenvalues_sorted[0] + lambda_3 * (rel_scale_fact - cov_ratio)

        lambdas = np.array([lambda_1, lambda_2, lambda_3])

        L = np.diag(lambdas[inverse_idxs]) * tot_scale_fact
    else:
        L = np.diag(eigenvalues) * tot_scale_fact

    Sigma = eigenvectors @ L @ eigenvectors.T

    cov[:3, :3] = Sigma

    return cov




def adjust_cov_quat(cov, tot_scale_fact=1.2, rel_scale_fact=0.15):
    
    cov_quat = cov[3:, 3:]

    eigenvalues, eigenvectors = np.linalg.eig(cov_quat)

    idxs = eigenvalues.argsort()
    inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
    for index, element in enumerate(idxs):
        inverse_idxs[element] = index

    eigenvalues_sorted  = np.sort(eigenvalues)
    cov_ratio = eigenvalues_sorted[2]/eigenvalues_sorted[3]
    if cov_ratio < rel_scale_fact:
        lambda_4 = eigenvalues_sorted[3]
        lambda_3 = eigenvalues_sorted[2] + lambda_4 * (rel_scale_fact - cov_ratio)
        lambda_2 = eigenvalues_sorted[1] + lambda_4 * (rel_scale_fact - cov_ratio)
        lambda_1 = eigenvalues_sorted[0] + lambda_4 * (rel_scale_fact - cov_ratio)

        lambdas = np.array([lambda_1, lambda_2, lambda_3, lambda_4])

        L = np.diag(lambdas[inverse_idxs]) * tot_scale_fact
    else:
        L = np.diag(eigenvalues) * tot_scale_fact


    Sigma = eigenvectors @ L @ eigenvectors.T

    cov[3:, 3:] = Sigma

    return cov
'''
