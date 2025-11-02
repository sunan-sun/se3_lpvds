import os
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R




def _process_bag(path):
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
        
        p_raw.append(pos_traj.T)
        q_raw.append([R.from_quat(quat_traj[:, i]) for i in range(quat_traj.shape[1]) ])
        t_raw.append(time_traj.reshape(time_traj.shape[1]))

    dt = (t_raw[0][-1] - t_raw[0][0])/len(t_raw[0])
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