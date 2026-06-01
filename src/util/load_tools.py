import os
import h5py
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R




def _process_bag(path, if_flip=False):
    """ Process .mat files that is converted from .bag files """

    data_ = loadmat(r"{}".format(path))
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    p_raw     = []
    q_raw     = []
    t_raw     = []

    sample_step = 5
    vel_thresh  = 1e-3 
    
    for l in range(L):
        data_l = data_[0, l]['pose'][0,0]
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]
        
        """ this is mostly for mani_py project"""
        if if_flip: 
            # Flip position trajectory
            pos_traj = np.fliplr(pos_traj)
            # Flip quaternion trajectory
            quat_traj = np.fliplr(quat_traj)
            # Flip and adjust time trajectory
            time_traj = np.fliplr(time_traj)
            time_traj = time_traj.max() - time_traj
        
        p_raw.append(pos_traj.T)
        q_raw.append([R.from_quat(quat_traj[:, i]) for i in range(quat_traj.shape[1]) ])
        t_raw.append(time_traj.reshape(time_traj.shape[1]))

    dt = np.average([t_raw[0][i+1] - t_raw[0][i] for i in range(len(t_raw[0])-1)])
    return p_raw, q_raw, t_raw, dt




def _get_sequence(seq_file):
    """
    Returns a list of containing each line of `seq_file`
    as an element

    Args:
        seq_file (str): File with name of demonstration files
                        in each line

    Returns:
        [str]: List of demonstration files
    """
    seq = None
    with open(seq_file) as x:
        seq = [line.strip() for line in x]
    return seq




def load_clfd_dataset(task_id=1, num_traj=1, sub_sample=3, duration=10.0):
    """
    Load data from clfd dataset

    Return:
    -------
        p_raw:  a LIST of L trajectories, each containing M observations of N dimension, or [M, N] ARRAY;
                M can vary and need not be same between trajectories

        q_raw:  a LIST of L trajectories, each containting a LIST of M (Scipy) Rotation objects;
                need to consistent with M from position
        
    Note:
    ----
        NO time stamp available in this dataset!

        [num_demos=9, trajectory_length=1000, data_dimension=7] 
        A data point consists of 7 elements: px,py,pz,qw,qx,qy,qz (3D position followed by quaternions in the scalar first format).
    """

    L = num_traj
    # T = 10.0            # pick a time duration 

    file_path           = os.path.dirname(os.path.realpath(__file__))  
    dir_path            = os.path.dirname(file_path)
    data_path           = os.path.dirname(dir_path)

    seq_file    = os.path.join(data_path, "dataset", "pos_ori", "robottasks_pos_ori_sequence_4.txt")
    filenames   = _get_sequence(seq_file)
    datafile    = os.path.join(data_path, "dataset", "pos_ori", filenames[task_id])
    
    data        = np.load(datafile)[:, ::sub_sample, :]

    p_raw = []
    q_raw = []
    t_raw = []

    for l in range(L):
        M = data[l, :, :].shape[0]

        data_ori = np.zeros((M, 4))         # convert to scalar last format, consistent with Scipy convention
        w        = data[l, :, 3 ].copy()  
        xyz      = data[l, :, 4:].copy()
        data_ori[:, -1]  = w
        data_ori[:, 0:3] = xyz

        p_raw.append(data[l, :, :3])
        q_raw.append([R.from_quat(q) for q in data_ori.tolist()])
        t_raw.append(np.linspace(0, duration, M, endpoint=False))   # hand engineer an equal-length time stamp

    dt = t_raw[0][1] - t_raw[0][0]
    return p_raw, q_raw, t_raw, dt




def load_demo_dataset():
    """
    Load demo data recorded from demonstration


    """

    input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "all.mat")
    
    return _process_bag(input_path)




def load_npy(duration):

    traj = np.load("dataset/UMI/traj1.npy")

    q_raw = [R.from_matrix(traj[i, :3, :3]) for i in range(traj.shape[0])]

    p_raw = [traj[i, :3, -1] for i in range(traj.shape[0])]

    """or provide T"""

    dt = duration/traj.shape[0]

    t_raw = [dt*i for i in range(traj.shape[0])]

    return [np.vstack(p_raw)], [q_raw], [t_raw], dt




#=======================================================================================
#=======================================================================================
#======= Codes below are designated for data collected using franka_interact============
#=======================================================================================
#=======================================================================================



def _truncate_stationary(ee_pos, timestamps, vel_threshold=0.02, pad=5,
                         ee_quat=None, ang_vel_threshold=0.05):
    """Trim leading and trailing samples where the EE is not moving.

    A sample is "moving" if its linear speed exceeds `vel_threshold` (m/s) OR,
    when `ee_quat` is given, its angular speed exceeds `ang_vel_threshold`
    (rad/s). Considering orientation matters for orientation-dominant rollouts
    (e.g. the modulated quaternion rollouts), whose position can reach the goal
    and freeze while the orientation is still settling — a position-only trim
    would chop off the orientation's return.
    """
    if len(ee_pos) < 3:
        return np.arange(len(ee_pos))

    dt = np.diff(timestamps)
    dt[dt <= 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1e-3
    speed = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1) / dt
    moving_mask = speed > vel_threshold

    if ee_quat is not None and len(ee_quat) == len(ee_pos):
        q = np.asarray(ee_quat, dtype=float)
        dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
        ang_speed = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0)) / dt
        moving_mask = moving_mask | (ang_speed > ang_vel_threshold)

    moving = np.where(moving_mask)[0]
    if moving.size == 0:
        return np.arange(len(ee_pos))

    start = max(moving[0] - pad, 0)
    end = min(moving[-1] + pad + 2, len(ee_pos))  # +2 to recover the diff offset
    return np.arange(start, end)



def _downsample(indices_len, target_length):
    if target_length is None or indices_len <= target_length:
        return np.arange(indices_len)
    return np.linspace(0, indices_len - 1, target_length).astype(int)




def load_franka_h5(file_path, vel_threshold=0.02, target_length=400,
                   fixed_time=False, T=4.0, pad=5, return_joints=False):
    """Load every demo from a franka_interact recording.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file (schema v2 with `demo_XXX` groups).
    vel_threshold : float
        Speed (m/s) below which a sample is considered stationary.
    target_length : int or None
        If set, each demo is uniformly downsampled to this many samples after
        truncation.
    fixed_time : bool
        If True, the returned timestamps are evenly spaced over `T` seconds
        instead of using the recorded `t_robot`.
    T : float
        Total trajectory duration used when `fixed_time=True`.
    pad : int
        Number of stationary samples kept around the moving segment.

    Returns
    -------
    p_list : list of (N, 3) np.ndarray
    q_list : list of list of scipy Rotation
    t_list : list of (N,) np.ndarray
    dt : float
        Mean step size of the last loaded demo (matches `load_h5`'s contract).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    p_list, q_list, t_list, qj_list = [], [], [], []
    dt = None

    with h5py.File(file_path, 'r') as hf:
        demo_keys = sorted(k for k in hf.keys() if k.startswith('demo_'))
        if not demo_keys:
            raise RuntimeError("No demo_* groups found in {}".format(file_path))

        for key in demo_keys:
            grp = hf[key]
            ee_pos = np.asarray(grp['ee_pos'])
            ee_quat = np.asarray(grp['ee_quat_xyzw'])
            t_robot = np.asarray(grp['t_robot'])
            q_joint = np.asarray(grp['q']) if return_joints else None

            # 1. Truncate the stationary head/tail (consider orientation too,
            #    so orientation-dominant rollouts aren't cut mid-return).
            keep = _truncate_stationary(ee_pos, t_robot,
                                        vel_threshold=vel_threshold, pad=pad,
                                        ee_quat=ee_quat)
            ee_pos = ee_pos[keep]
            ee_quat = ee_quat[keep]
            t_robot = t_robot[keep]
            if q_joint is not None:
                q_joint = q_joint[keep]

            # 2. Optionally downsample to a fixed length.
            sub = _downsample(len(ee_pos), target_length)
            ee_pos = ee_pos[sub]
            ee_quat = ee_quat[sub]
            t_robot = t_robot[sub]
            if q_joint is not None:
                q_joint = q_joint[sub]

            traj_len = len(ee_pos)
            if traj_len < 2:
                continue

            # 3. Build the trajectory in the se3_lpvds format.
            q_raw = [R.from_quat(ee_quat[i]) for i in range(traj_len)]
            p_raw = ee_pos.astype(np.float64)

            if fixed_time:
                dt_local = T / traj_len
                t_raw = np.array([dt_local * i for i in range(traj_len)])
            else:
                t_raw = t_robot - t_robot[0]
                dt_local = float(np.mean(np.diff(t_raw)))

            p_list.append(p_raw)
            q_list.append(q_raw)
            t_list.append(t_raw)
            if q_joint is not None:
                qj_list.append(q_joint.astype(np.float64))
            dt = dt_local

    if not p_list:
        raise RuntimeError("All demos in {} were empty after truncation".format(file_path))

    if return_joints:
        return p_list, q_list, t_list, dt, qj_list
    return p_list, q_list, t_list, dt