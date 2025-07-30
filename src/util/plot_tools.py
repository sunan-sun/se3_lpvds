import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
import random


font = {'family' : 'Times New Roman',
         'size'   : 18
         }
mpl.rc('font', **font)






def plot_vel(v_test, w_test):
    """Plots the linear and angular velocities over time.

    Args:
        v_test: List or array of linear velocities (M x 3).
        w_test: List or array of angular velocities (M x 3).
    """
    v_test = np.vstack(v_test)
    M, N = v_test.shape

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(3):
        axs[k].scatter(np.arange(M), v_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])
    
    axs[0].set_title("Linear Velocity")

    w_test = np.vstack(w_test)
    M, N = w_test.shape
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(3):
        axs[k].scatter(np.arange(M), w_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])
    axs[0].set_title("Angular Velocity")



def plot_gamma(gamma_arr, **argv):
    """Plots the gamma values (activation weights) over time for each component.

    Args:
        gamma_arr: Array of gamma values (M x K), where M is the number of time steps
                   and K is the number of components.
        **argv: Additional keyword arguments. Can include "title" for the plot title.
    """

    M, K = gamma_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(M), gamma_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])
    else:
        axs[0].set_title(r"$\gamma(\cdot)$ over Time")




def plot_result(p_train, p_test, q_test):
    """Plots the 3D trajectory reproduction results.

    Displays the demonstrated trajectory, the reproduced trajectory,
    initial/target points, and orientation frames along the reproduced path.

    Args:
        p_train: Demonstrated trajectory positions (N x 3).
        p_test: Reproduced trajectory positions (M x 3).
        q_test: Reproduced trajectory orientations as SciPy Rotation objects (list or array of length M).
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')


    ax.plot(p_train[:, 0], p_train[:, 1], p_train[:, 2], 'o', color='gray', alpha=0.2, markersize=1.5, label="Demonstration")
    ax.plot(p_test[:, 0], p_test[:, 1], p_test[:, 2],  color='k', label = "Reproduction")

    ax.scatter(p_test[0, 0], p_test[0, 1], p_test[0, 2], 'o', facecolors='none', edgecolors='magenta',linewidth=2,  s=100, label="Initial")
    ax.scatter(p_test[-1, 0], p_test[-1, 1], p_test[-1, 2], marker=(8, 2, 0), color='k',  s=100, label="Target")


    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    
    for i in np.linspace(0, len(q_test), num=30, endpoint=False, dtype=int):

        r = q_test[i]
        loc = p_test[i, :]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    ax.axis('equal')
    ax.legend(ncol=4, loc="upper center")

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))





def plot_gmm(p_in, gmm):
    """Plots the GMM clustering results in 3D.

    Displays the input data points colored by their assigned cluster,
    and shows the mean orientation of each Gaussian component.

    Args:
        p_in: Input position data (N x 3).
        gmm: Fitted GMM object, expected to have attributes 'assignment_arr' (N,),
             'K' (number of components), and 'gaussian_list' (list of dicts,
             where each dict contains ["mu"][1] as the mean orientation).
    """

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    for k in range(K):
        label_k =np.where(label == k)[0]

        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)

        r = gmm.gaussian_list[k]["mu"]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    ax.axis('equal')


    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))






def plot_gmm_pos(p_in, gmm):

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB


    ax.axis('equal')


    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))




def plot_p_out(p_in, p_out, pos_obj):
    """compare estimated p_out given p_in with groud truth"""


    p_out_est     = np.zeros((p_out.shape)) # (M, N)

    gamma = pos_obj.damm.logProb(p_in)

    for k in range(pos_obj.K):
        p_out_est  +=  (np.tile(gamma[k, :], (3, 1)) * (pos_obj.A[k] @ (p_in - pos_obj.x_att.reshape(1, -1)).T)).T


    M, N = p_out.shape

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    label = pos_obj.assignment_arr

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    label_list =["v_x", "v_y", "v_z"]
    for k in range(3):
        axs[k].scatter(np.arange(M), p_out[:, k], s=5, color=color_mapping, label=label_list[k])
        axs[k].scatter(np.arange(M), p_out_est[:, k], s=5, color=color_mapping, alpha=0.2)




def plot_q_out(q_in, q_out, ori_obj):
    """compare estimated p_out given p_in with groud truth"""

    q_out = list_to_arr(q_out)

    q_out_est     = np.zeros((q_out.shape)) # (M, N)

    gamma = ori_obj.gmm.logProb(q_in)

    q_diff  = riem_log(ori_obj.q_att, q_in)
    
    q_out_att     = np.zeros((q_out.shape)) # (M, N)
    for k in range(ori_obj.K):
        q_out_att  +=  (np.tile(gamma[k, :], (4, 1)) * (ori_obj.A_ori[k] @ q_diff.T)).T


    for i in range(q_out_est.shape[0]):
        q_out_body = parallel_transport(ori_obj.q_att, q_in[i], q_out_att[i, :].T)
        q_out_q    = riem_exp(q_in[i], q_out_body) 
        # q_out_i      = R.from_quat(q_out_q.reshape(4,))
        q_out_est[i, :] = q_out_q




    M, N = q_out_est.shape

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    label = ori_obj.gmm.assignment_arr

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    label_list =["q_x", "q_y", "q_z", "q_w"]
    for k in range(4):
        axs[k].scatter(np.arange(M), q_out[:, k], s=5, color=color_mapping, label=label_list[k])
        axs[k].scatter(np.arange(M), q_out_est[:, k], s=5, color=color_mapping, alpha=0.2)

    
