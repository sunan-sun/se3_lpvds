"""
Utility functions for processing and manipulating trajectories in SE(3) space.
This module provides tools for:
- Shifting, smoothing, and filtering position and orientation trajectories
- Computing velocities from trajectory data
- Managing quaternions for orientation representation
- Preprocessing trajectory data for dynamical systems learning
- Data augmentation through interpolation to increase trajectory density
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from . import quat_tools
from .quat_tools import *




def _compute_ang_vel(q_i, q_ip1, dt=0.01):
    """  
    Compute angular velocity between two consecutive orientations
    
    Args:
        q_i: Current quaternion (as scipy Rotation)
        q_ip1: Next quaternion
        dt: Time step between the orientations
        
    Returns:
        w: Angular velocity vector
    """

    # dq = q_i.inv() * q_ip1    # from q_i to q_ip1 in body frame
    dq = q_ip1 * q_i.inv()    # from q_i to q_ip1 in fixed frame

    dq = dq.as_rotvec() 
    w  = dq / dt

    return w




def _shift_pos(p_list):
    """
    Shift position trajectories so they end at the same point
    
    Args:
        p_list: List of position trajectories
        
    Returns:
        p_shifted: List of shifted position trajectories
        p_att_mean: Mean attractor position (final position)
    """

    L = len(p_list)

    p_att_list  = [p_list[l][-1, :]  for l in range(L)]  
    p_att_mean  =  np.mean(np.array(p_att_list), axis=0)

    p_shifted = []
    for l in range(L):
        p_diff = p_att_mean - p_att_list[l]

        p_shifted.append(p_diff.reshape(1, -1) + p_list[l])

    return p_shifted, p_att_mean, p_att_list


def _shift_ori(q_list):
    """
    Shift orientation trajectories so they end at the same orientation
    
    Args:
        q_list: List of orientation trajectories (quaternions)
        
    Returns:
        q_shifted: List of shifted orientation trajectories
        q_att_mean: Mean attractor orientation (final orientation)
    
    Note:
    ---- 
        Scipy methods, e.g. "R.mean()", "R.inv()" and "R.__mul__()" will OFTEN flip the SIGNS of the computed quaternion
        
        Do NOT output "q_att_mean" as the ATTRACTOR which could be SIGN-inconsistent with the rest of quaternions
        INSTEAD, always output the LAST of the shifted quaternions
    """

    L = len(q_list)

    q_att_list  = [q_list[l][-1]  for l in range(L)]                      
    q_att_mean  = R.from_quat([q_att_list[l].as_quat() for l in range(L)]).mean()

    q_shifted = []
    for l in range(L):
        q_diff = q_att_mean * q_att_list[l].inv()

        q_shifted.append([q_diff * q for q in q_list[l]])

    return q_shifted, q_shifted[-1][-1], q_att_list


def _smooth_pos(p_in:list, k=10):
    """ 
    Smooth position trajectories using Savitzky-Golay filter
    
    Args:
        p_in: List of position trajectories
        k: Window length for filter
        
    Returns:
        p_smooth: List of smoothed position trajectories
    """
    p_smooth = []
    for l in range(len(p_in)):
        p_smooth.append(savgol_filter(p_in[l], window_length=k, polyorder=2, axis=0, mode="nearest"))
    
    return p_smooth





def _smooth_ori(q_list, q_att, opt):
    """
    Smoothen the orientation trajectory using Savgol filter or SLERP interpolation
    
    Args:
        q_list: List of orientation trajectories
        q_att: Attractor orientation
        opt: Smoothing method ("savgol" or "slerp")
        
    Returns:
        q_smooth: List of smoothed orientation trajectories

    Note:
    ----
        The value of k are parameters that can be tuned in both methods
    """

    L = len(q_list)

    q_smooth = []
    for l in range(L):
        q_l    = q_list[l]

        if opt == "savgol":
            k =  80

            q_l_att  = quat_tools.riem_log(q_att, q_l)

            q_smooth_att = savgol_filter(q_l_att, window_length=k, polyorder=2, axis=0, mode="nearest")

            q_smooth_arr = quat_tools.riem_exp(q_att, q_smooth_att)

            q_smooth.append([R.from_quat(q_smooth_arr[i, :]) for i in range(q_smooth_arr.shape[0])])
    
    
        elif opt == "slerp":
            k = 80

            t_list = [0.1*i for i in range(len(q_l))]
            
            idx_list  = np.linspace(0, len(q_l)-1, num=int(len(q_l)/k), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]
            key_rots  = R.from_quat([q_l[i].as_quat() for i in idx_list])
            
            slerp = Slerp(key_times, key_rots)

            idx_list  = np.linspace(0, len(q_l)-1, num=int(len(q_l)), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]

            q_interp  = slerp(key_times)
            q_smooth.append([q_interp[i] for i in range(len(q_interp))])

    return q_smooth




def _filter(p_list, q_list, t_list):
    """   
    Extract a smooth velocity profile by filtering orientation trajectories
    Only keeps points with sufficient angular velocity until near the attractor
    
    Args:
        p_list: List of position trajectories
        q_list: List of orientation trajectories
        t_list: List of time stamps
        
    Returns:
        p_filter: Filtered position trajectories
        q_filter: Filtered orientation trajectories
        t_filter: Filtered time stamps
    """

    min_thold = 0.05  # Minimum angular velocity threshold
    pct_thold = 0.8   # Percentage of trajectory to apply filtering to

    L = len(q_list)
    
    p_filter  = []
    q_filter  = []
    t_filter  = []

    for l in range(L):
        M       = len(q_list[l])
        M_thold = M * pct_thold

        p_filter_l  = [p_list[l][0, :]]
        q_filter_l  = [q_list[l][0]]
        t_filter_l  = [t_list[l][0]]

        for i in range(M-1):
            q_i    = q_filter_l[-1]
            q_ip1  = q_list[l][i+1]
            dt     = t_list[l][i+1] - t_list[l][i]
            w      = _compute_ang_vel(q_i, q_ip1, dt)

            if (i<=M_thold and np.linalg.norm(w)>=min_thold) or i>M_thold:  
                p_filter_l.append(p_list[l][i+1, :])
                q_filter_l.append(q_list[l][i+1])
                t_filter_l.append(t_list[l][i+1])

        p_filter.append(np.array(p_filter_l))
        q_filter.append(q_filter_l)
        t_filter.append(np.array(t_filter_l))

    return p_filter, q_filter, t_filter


def pre_process(p_raw, q_raw, t_raw, shift=False, opt="savgol"):
    """
    Complete preprocessing pipeline for position and orientation trajectories
    
    Args:
        p_raw: Raw position trajectories
        q_raw: Raw orientation trajectories
        t_raw: Raw time stamps
        opt: Smoothing method for orientations
        
    Returns:
        p_in: Preprocessed position trajectories
        q_in: Preprocessed orientation trajectories
        t_raw: Time stamps
    """
    if shift:
        p_in, p_att, p_att_list = _shift_pos(p_raw)
        q_in, q_att, q_att_list = _shift_ori(q_raw)
        p_in = _smooth_pos(p_in)
    else:
        L = len(p_raw)
        p_att_list  = [p_raw[l][-1, :]  for l in range(L)]  
        p_att  =  np.mean(np.array(p_att_list), axis=0)

        q_att_list  = [q_raw[l][-1]  for l in range(L)]  
        q_att  = R.from_quat([q_att_list[l].as_quat() for l in range(L)]).mean()

        p_in                    = _smooth_pos(p_raw)
        q_in                    = q_raw
    # q_in                    = _smooth_ori(q_in, q_att, opt) # needed or not?

    # p_in, q_in, t_in        = _filter(p_in, q_in, t_raw)  # needed or not?

    return p_in, q_in, t_raw, p_att, q_att


def compute_output(p_list, q_list, t_list):
    """
    Compute velocities from position and orientation trajectories
    
    Args:
        p_list: List of position trajectories
        q_list: List of orientation trajectories
        t_list: List of time stamps
        
    Returns:
        p_out: List of position velocities
        q_out: List of orientations (for each velocity point)
    """

    L = len(q_list)

    p_out = []
    q_out = []

    for l in range(L):
        M       = len(q_list[l])
        if M == 1:
            p_out_l = [np.zeros((3))]
            q_out_l = [q_list[l][0]]
            p_out.append(np.array(p_out_l))
            q_out.append(q_out_l)
            continue

        p_out_l  = []
        q_out_l  = []
        
        
        for i in range(M-1):
            p_i    = p_list[l][i, :]
            p_ip1  = p_list[l][i+1, :]

            q_i    = q_list[l][i]
            q_ip1  = q_list[l][i+1]

            dt     = t_list[l][i+1] - t_list[l][i]

            v      = (p_ip1 - p_i) / dt

            p_out_l.append(v)
            q_out_l.append(q_ip1)

        p_out_l.append(v)  # Repeat the last velocity
        q_out_l.append(q_ip1)  # Add the last orientation

        p_out.append(np.array(p_out_l))
        q_out.append(q_out_l)

    return p_out, q_out




def extract_state(p_list, q_list):
    """
    Extract initial states and attractor state from trajectories
    
    Args:
        p_list: List of position trajectories
        q_list: List of orientation trajectories
        
    Returns:
        p_init: List of initial positions
        q_init: List of initial orientations
        p_att: Attractor position
        q_att: Attractor orientation
    """
    L = len(q_list)

    p_init = []  # list of L initial points given L trajectories
    q_init = []
    
    for l in range(L):
        p_init.append(p_list[l][0, :])
        q_init.append(q_list[l][0])

    p_att = p_list[0][-1, :]
    q_att = q_list[0][-1]

    return p_init, q_init, p_att, q_att




def rollout_list(p_in, q_in, p_out, q_out):
    """ 
    Roll out the nested list of trajectories into single arrays
    
    Args:
        p_in: List of position trajectories
        q_in: List of orientation trajectories
        p_out: List of position velocities
        q_out: List of orientation trajectories for velocities
        
    Returns:
        p_in_rollout: Flattened array of positions
        q_in_rollout: Flattened list of orientations
        p_out_rollout: Flattened array of velocities
        q_out_rollout: Flattened list of orientations for velocities
    """

    L = len(q_in)

    for l in range(L):
        if l == 0:
            p_in_rollout = p_in[l]
            q_in_rollout = q_in[l]
            p_out_rollout = p_out[l]
            q_out_rollout = q_out[l]
        else:
            p_in_rollout = np.vstack((p_in_rollout, p_in[l]))
            q_in_rollout += q_in[l]
            p_out_rollout = np.vstack((p_out_rollout, p_out[l]))
            q_out_rollout += q_out[l]
            

    return p_in_rollout, q_in_rollout, p_out_rollout, q_out_rollout


def _interpolate_positions(p_traj, t_traj, factor=3):
    """
    Interpolate position trajectory to create more data points
    
    Args:
        p_traj: Position trajectory array of shape (N, 3)
        t_traj: Time array of shape (N,)
        factor: Interpolation factor (3 means 3x more points)
        
    Returns:
        p_interp: Interpolated position trajectory
        t_interp: Interpolated time array
    """
    N = len(t_traj)
    
    # Create new time points with factor times more resolution
    t_new = np.linspace(t_traj[0], t_traj[-1], (N-1) * factor + 1)
    
    # Interpolate each dimension separately
    p_interp = np.zeros((len(t_new), 3))
    for dim in range(3):
        f = interp1d(t_traj, p_traj[:, dim], kind='linear')
        p_interp[:, dim] = f(t_new)
    
    return p_interp, t_new


def _interpolate_orientations(q_traj, t_traj, factor=3):
    """
    Interpolate orientation trajectory using SLERP to create more data points
    
    Args:
        q_traj: List of scipy Rotation objects
        t_traj: Time array of shape (N,)
        factor: Interpolation factor (3 means 3x more points)
        
    Returns:
        q_interp: List of interpolated scipy Rotation objects
        t_interp: Interpolated time array
    """
    N = len(t_traj)
    
    # Create new time points with factor times more resolution
    t_new = np.linspace(t_traj[0], t_traj[-1], (N-1) * factor + 1)
    
    # Convert list of Rotation objects to a single Rotation object for SLERP
    q_array = R.from_quat([q.as_quat() for q in q_traj])
    
    # Create SLERP interpolator
    slerp = Slerp(t_traj, q_array)
    
    # Interpolate orientations
    q_interp_array = slerp(t_new)
    
    # Convert back to list of individual Rotation objects
    q_interp = [R.from_quat(q_interp_array[i].as_quat()) for i in range(len(q_interp_array))]
    
    return q_interp, t_new


def augment_data(p_list, q_list, t_list, factor=3):
    """
    Augment trajectory data by interpolating between points to create more data points
    
    Args:
        p_list: List of position trajectories (each is numpy array of shape (N, 3))
        q_list: List of orientation trajectories (each is list of scipy Rotation objects)
        t_list: List of time arrays (each is numpy array of shape (N,))
        factor: Interpolation factor (3 means 3x more points between each pair of original points)
        
    Returns:
        p_augmented: List of augmented position trajectories
        q_augmented: List of augmented orientation trajectories  
        t_augmented: List of augmented time arrays
    """
    L = len(p_list)
    
    p_augmented = []
    q_augmented = []
    t_augmented = []
    
    for l in range(L):
        # Skip trajectories that are too short to interpolate
        if len(p_list[l]) < 2:
            p_augmented.append(p_list[l])
            q_augmented.append(q_list[l])
            t_augmented.append(t_list[l])
            continue
            
        # Interpolate positions
        p_interp, t_interp = _interpolate_positions(p_list[l], t_list[l], factor)
        
        # Interpolate orientations
        q_interp, _ = _interpolate_orientations(q_list[l], t_list[l], factor)
        
        p_augmented.append(p_interp)
        q_augmented.append(q_interp)
        t_augmented.append(t_interp)
    
    return p_augmented, q_augmented, t_augmented


def preprocess_with_augmentation(p_raw, q_raw, t_raw, augment_factor=6, shift=False, opt="savgol"):
    """
    Complete preprocessing pipeline with data augmentation
    
    Args:
        p_raw: Raw position trajectories
        q_raw: Raw orientation trajectories  
        t_raw: Raw time stamps
        augment_factor: Factor for data augmentation (3 means 3x more points)
        shift: Whether to shift trajectories to common attractor
        opt: Smoothing method for orientations
        
    Returns:
        p_in: Preprocessed and augmented position trajectories
        q_in: Preprocessed and augmented orientation trajectories
        t_in: Augmented time stamps
        p_att: Attractor position
        q_att: Attractor orientation
    """
    # First augment the data
    p_aug, q_aug, t_aug = augment_data(p_raw, q_raw, t_raw, factor=augment_factor)
    
    # Then apply standard preprocessing
    p_in, q_in, t_in, p_att, q_att = pre_process(p_aug, q_aug, t_aug, shift=shift, opt=opt)
    
    return p_in, q_in, t_in, p_att, q_att


