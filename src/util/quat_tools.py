import sys
import numpy as np
from scipy.spatial.transform import Rotation as R


"""
@note all operations below, of which the return is a vector, return 1-D array, 
      unless multiple inputs are given in vectorized operations
"""


def quat_mean(q_list):
    """
    Given a list of R objects, compute quaternion average while retaining the proper sign
    """

    q_avg = q_list[0]
    errors = np.zeros((3, len(q_list)))
    error_sum = np.ones(3)
    while np.linalg.norm(error_sum) > 0.01:
        for idx, q  in enumerate(q_list):
            error = q * q_avg.inv()
            errors[:, idx] = error.as_rotvec()
        error_sum = np.mean(errors, 1)
        q_err  = R.from_rotvec(error_sum)
        q_avg = q_err * q_avg

    return q_avg


def _process_x(x):
    """
    x can be either
        - a single R object
        - a list of R objects
    """

    if isinstance(x, list):
        x = list_to_arr(x)
    elif isinstance(x, R):
        x = x.as_quat()[np.newaxis, :]

    return x



def _process_xy(x, y):
    """
    Transform both x and y into (N by M) np.ndarray and normalize to ensure unit quaternions

    x and y can be either
        - 2 single R objects
        - 1 single R object + 1 list of R objects
        - 2 lists of R objects
    
    Except when both x and y are single R objects, always expand and cast the single R object to meet the same shape
    """
    
    M = 4
    if isinstance(x, R) and isinstance(y, list):
        N = len(y)
        x = np.tile(x.as_quat()[np.newaxis, :], (N,1))
        y = list_to_arr(y)

    elif isinstance(y, R) and isinstance(x, list):
        N = len(x)
        y = np.tile(y.as_quat()[np.newaxis, :], (N,1))
        x = list_to_arr(x)

    elif isinstance(x, list) and isinstance(y, list):
        x = list_to_arr(x)
        y = list_to_arr(y)
    
    elif isinstance(x, R) and isinstance(y, R):
        if x.as_quat().ndim == 1:
            x = x.as_quat()[np.newaxis, :]
        else:
            x = x.as_quat()
        if y.as_quat().ndim == 1:
            y = y.as_quat()[np.newaxis, :]
        else:
            y = y.as_quat()

    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.ndim == 1 and y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
        M = x.shape[1]

    else:
        print("Invalid inputs in quaternion operation")
        sys.exit()

    x = x / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,M))
    y = y / np.tile(np.linalg.norm(x, axis=1, keepdims=True), (1,M))
    
    # --------------------------------------------------------------------
    # Canonicalize quaternion signs to remove the double-cover ambiguity.
    # For each pair (x_i, y_i) make sure they lie in the same hemisphere of
    # the 4-D unit sphere.  If the dot product is negative, flip the sign of
    # y_i so that dot(x_i, y_i) ≥ 0.  This guarantees that subsequent
    # geodesic computations (angle, log map, etc.) follow the shortest arc
    # and prevents spurious equilibria at ~180° from the intended
    # orientation.
    # --------------------------------------------------------------------

    dot_xy = np.sum(x * y, axis=1, keepdims=True)  # shape (N,1)
    sign_correction = np.where(dot_xy < 0.0, -1.0, 1.0)  # (N,1)
    y = y * sign_correction  # broadcast sign flip where needed

    # Re-normalise (just in case) after sign flip
    y = y / np.tile(np.linalg.norm(y, axis=1, keepdims=True), (1, M))
    
    return x,y




def unsigned_angle(x, y):
    """
    Vectorized operation

    @param x is always a 1D array
    @param y is either a 1D array or 2D array of N by M

    note: "If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b; i.e. sum(a[i,:] * b) "
    note: "/" divide operator equivalent to np.divide, performing element-wise division
    note:  np.dot, np.linalg.norm(keepdims=False) and the return angle are 1-D array
    """
    x, y = _process_xy(x, y)

    dotProduct = np.sum(x * y, axis=1)

    angle = np.arccos(np.clip(dotProduct, -1, 1))

    return angle





def riem_log(x, y):
    """
    Vectorized operation

    @param x is the point of tangency
    @param y is either a 1D array or 2D array of N by M


    @note special cases to take care of when x=y and angle(x, y) = pi
    @note IF further normalization needed after adding perturbation?

    - Scenario 1:
        When projecting q_train wrt q_att:
            x is a single R object
            y is a list of R objects
    
    - Scenario 2:
        When projecting each w_train wrt each q_train:
            x is a list of R objects
            y is a list of R objects
    
    - Scenario 3:
        When parallel_transport each projected w_train from respective q_train to q_att:
            x is a list of R objects
            y is a single R object

    - Scenario 4:
        When simulating forward, projecting q_curr wrt q_att:
            x is a single R object
            y is a single R object
    """

    np.seterr(invalid='ignore')

    x, y = _process_xy(x, y)

    N, M = x.shape

    angle = unsigned_angle(x, y) 

    y[angle == np.pi] += 0.001 

    x_T_y = np.tile(np.sum(x * y, axis=1,keepdims=True), (1,M))

    x_T_y_x = x_T_y * x

    u_sca =  np.tile(angle[:, np.newaxis], (1, M))
    u_vec =  (y-x_T_y_x) / np.tile(np.linalg.norm(y-x_T_y_x, axis=1, keepdims=True), (1, M))

    u  = u_sca * u_vec

    
    """
    When y=x, the u should be 0 instead of nan.
    Either of the methods below would work
    """
    # u[np.isnan(u)] = 0
    u[angle == 0] = np.zeros([1, M]) 

    return u


def parallel_transport(x, y, v):
    """
    Vectorized operation
    
    parallel transport a vector u from space defined by x to a new space defined by y

    @param: x original tangent point
    @param: y new tangent point
    @param v vector in tangent space (compatible with both 1-D and 2-D NxM)

    """
    v = _process_x(v)
    log_xy = riem_log(x, y)
    log_yx = riem_log(y, x)
    d_xy = unsigned_angle(x, y)


    # a = np.sum(log_xy * v, axis=1) 
    u = v - (log_xy + log_yx) * np.tile(np.sum(log_xy * v, axis=1, keepdims=True) / np.power(d_xy,2)[:, np.newaxis], (1, 4))


    # Find rows containing NaN values
    nan_rows = np.isnan(u).all(axis=1)

    # Replace NaN rows with zero vectors
    u[nan_rows, :] = np.zeros((1, 4))
 
    return u


def riem_exp(x, v):
    """
    Used during 
         i) running savgol filter
        ii) simulation where x is a rotation object, v is a numpy array
    """

    x = _process_x(x)

    if v.shape[0] == 1:

        v_norm = np.linalg.norm(v)

        if v_norm == 0:
            return x

        y = x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)
    
    else:
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)

        y = np.tile(x, (v_norm.shape[0], 1)) * np.tile(np.cos(v_norm), (1,4)) + v / np.tile(v_norm / np.sin(v_norm), (1,4)) 


    # # Find rows containing NaN values
    # nan_rows = np.isnan(y).all(axis=1)

    # # Replace NaN rows with zero vectors
    # y[nan_rows, :] = np.zeros((1, 4))

    return y





def riem_cov(q_mean, q_list):

    q_list_mean = riem_log(q_mean, q_list)
    scatter = q_list_mean.T @ q_list_mean


    cov = scatter/len(q_list)


    return cov




def canonical_quat(q):
    """
    Force all quaternions to have positive scalar part; necessary to ensure proper propagation in DS
    """
    if (q[-1] < 0):
        return -q
    else:
        return q
    


def list_to_arr(q_list):

    N = len(q_list)
    M = 4

    q_arr = np.zeros((N, M))

    for i in range(N):
        q_arr[i, :] = q_list[i].as_quat()

        # q_arr[i, :] = canonical_quat(q_list[i].as_quat())

    return q_arr


def list_to_euler(q_list):

    N = len(q_list)
    M = 3

    q_arr = np.zeros((N, M))

    for i in range(N):
        q_arr[i, :] = q_list[i].as_euler('xyz')

        # q_arr[i, :] = canonical_quat(q_list[i].as_quat())

    return q_arr
